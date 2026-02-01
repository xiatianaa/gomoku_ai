import copy
import logging
import math
import time
from contextlib import nullcontext

import numpy as np
import torch


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

BOARD_SIZE = 15
DEFAULT_SIMULATIONS = 1500


def get_adaptive_simulations(board_state):
    """根据棋盘阶段动态调整搜索次数。"""
    piece_count = sum(row.count(1) + row.count(2) for row in board_state)

    if piece_count < 10:
        return 1500
    if piece_count < 25:
        return 1800
    return 2000


def _load_model_weights(model, model_path, device):
    """加载模型权重，兼容多种保存格式。"""
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)


def _device_supports_fp16(device):
    """检测当前设备是否支持FP16混合精度。"""
    if device.type != "cuda":
        return False
    try:
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(device_index)
    except Exception:
        return False
    return (major, minor) >= (5, 3)


class ValueCNN(torch.nn.Module):
    """五子棋价值网络模型。"""

    def __init__(self, in_channels=3, hidden_channels=32, num_blocks=5, value_dim=128):
        super(ValueCNN, self).__init__()

        self.conv_init = torch.nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn_init = torch.nn.BatchNorm2d(hidden_channels)

        self.res_blocks = torch.nn.ModuleList([
            self.ResidualBlock(hidden_channels) for _ in range(num_blocks)
        ])

        self.policy_conv1 = torch.nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1)
        self.policy_bn1 = torch.nn.BatchNorm2d(hidden_channels // 2)
        self.policy_conv2 = torch.nn.Conv2d(hidden_channels // 2, 1, kernel_size=3, padding=1)

        self.value_conv = torch.nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.value_bn = torch.nn.BatchNorm2d(1)
        self.value_fc1 = torch.nn.Linear(BOARD_SIZE * BOARD_SIZE, value_dim)
        self.value_fc2 = torch.nn.Linear(value_dim, 1)

    class ResidualBlock(torch.nn.Module):
        def __init__(self, channels):
            super(ValueCNN.ResidualBlock, self).__init__()
            self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.bn1 = torch.nn.BatchNorm2d(channels)
            self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.bn2 = torch.nn.BatchNorm2d(channels)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            residual = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += residual
            out = self.relu(out)
            return out

    def forward(self, x):
        x = torch.nn.functional.relu(self.bn_init(self.conv_init(x)))

        for block in self.res_blocks:
            x = block(x)

        policy = torch.nn.functional.relu(self.policy_bn1(self.policy_conv1(x)))
        policy = self.policy_conv2(policy)
        policy = policy.squeeze(1)
        policy_logits = policy.view(x.size(0), -1)

        value = torch.nn.functional.relu(self.value_bn(self.value_conv(x)))
        value = value.view(x.size(0), -1)
        value = torch.nn.functional.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return value, policy_logits

    def calc(self, x):
        self.eval()
        with torch.no_grad():
            value, logits = self.forward(x)
            probs = torch.nn.functional.softmax(logits, dim=1).view(-1, BOARD_SIZE, BOARD_SIZE)
            return value, probs


class MCTSNode:
    """MCTS节点。"""

    def __init__(self, board, parent=None, move=None):
        self.board = board
        self.parent = parent
        self.move = move
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.val = 0
        self.value = None

    def update_value(self):
        if len(self.children) == 0:
            self.val = self.value
        else:
            self.val = self.value_sum / self.visit_count


class GomokuAI:
    """五子棋AI决策器。"""

    def __init__(
        self,
        model_path,
        device=None,
        simulations=DEFAULT_SIMULATIONS,
        c_puct=1.0,
        use_rand=0.01,
        adaptive_simulations=False,
        use_mixed_precision=None,
    ):
        self.base_simulations = simulations
        self.adaptive_simulations = adaptive_simulations
        self.c_puct = c_puct
        self.use_rand = use_rand

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(resolved_device)
        self.model = ValueCNN()
        _load_model_weights(self.model, model_path, self.device)
        self.model.to(self.device)
        self.model.eval()

        fp16_supported = _device_supports_fp16(self.device)
        if use_mixed_precision is None:
            self.use_mixed_precision = fp16_supported
        elif use_mixed_precision:
            self.use_mixed_precision = fp16_supported
            if not fp16_supported:
                logger.warning("当前GPU不支持FP16，混合精度推理已自动关闭。")
        else:
            self.use_mixed_precision = False
            if self.device.type == "cuda":
                logger.info("已按配置在CUDA设备上禁用混合精度推理。")

        logger.info(
            "GomokuAI 初始化完成，使用设备: %s | 模拟次数: %s | 自适应: %s | 混合精度: %s",
            self.device.type,
            self.base_simulations,
            self.adaptive_simulations,
            self.use_mixed_precision,
        )

    def board_to_tensor(self, board):
        board_array = np.array(board)
        current_player = (board_array == 1).astype(np.float32)
        opponent = (board_array == -1).astype(np.float32)
        empty = (board_array == 0).astype(np.float32)
        tensor = np.stack([current_player, opponent, empty], axis=0)
        return torch.FloatTensor(tensor)

    @staticmethod
    def convert_board_format(board_state, player):
        converted = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                cell = board_state[i][j]
                if cell == 0:
                    converted[i][j] = 0
                elif cell == player:
                    converted[i][j] = 1
                else:
                    converted[i][j] = -1

        return converted

    def evaluation_func(self, board):
        num_used = sum(1 for i in range(BOARD_SIZE) for j in range(BOARD_SIZE) if board[i][j] != 0)

        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board[i][j] != 0:
                    for (x, y) in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                        cnt = 0
                        for d in range(5):
                            ni = i + d * x
                            nj = j + d * y
                            if (
                                0 <= ni < BOARD_SIZE
                                and 0 <= nj < BOARD_SIZE
                                and board[i][j] == board[ni][nj]
                            ):
                                cnt += 1
                            else:
                                break
                        if cnt == 5:
                            score = 1 - num_used * 3e-4
                            return score if board[i][j] == 1 else -score
        return 0

    @staticmethod
    def no_child(board):
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board[i][j] == 0:
                    return False
        return True

    def is_terminal(self, board):
        if self.no_child(board):
            return True
        return self.evaluation_func(board) != 0

    def get_calc(self, board):
        board_tensor = self.board_to_tensor(board).unsqueeze(0).to(self.device)
        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.use_mixed_precision
            else nullcontext()
        )
        with torch.no_grad():
            with amp_context:
                value, policy = self.model.calc(board_tensor)
        return float(value.float()), policy.squeeze(0).cpu().numpy().tolist()

    def expand_node(self, node):
        node.value, policy = self.get_calc(node.board)

        sum_1 = 0
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if node.board[i][j] == 0:
                    sum_1 += policy[i][j]
        if sum_1 == 0:
            sum_1 += 1e-10

        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if node.board[i][j] == 0:
                    node.children[(i, j)] = None, policy[i][j] / sum_1 + np.random.normal(0, self.use_rand)

    def select_child(self, node):
        total_visits = sum((child.visit_count if child is not None else 0) for child, _ in node.children.values())
        explore_buff = math.pow(total_visits + 1, 0.5)

        exp1 = 0
        exp2 = 0
        for child, _ in node.children.values():
            if child is not None:
                exp1 += child.val * child.visit_count
                exp2 += child.visit_count
        ave = exp1 / (exp2 + 1e-5)

        best_score = -1e9
        best_move = None

        for move, (child, prior) in node.children.items():
            explore = self.c_puct * prior * explore_buff
            exploit = ave
            if child is not None and child.visit_count != 0:
                exploit = child.val
                explore /= (child.visit_count + 1)

            score = explore - exploit
            if score > best_score:
                best_score = score
                best_move = move

        chd, pri = node.children[best_move]
        if chd is None:
            i, j = best_move
            new_board = copy.deepcopy(node.board)
            new_board[i][j] = 1
            for x in range(BOARD_SIZE):
                for y in range(BOARD_SIZE):
                    new_board[x][y] *= -1
            chd = MCTSNode(new_board, parent=node, move=best_move)
            node.children[best_move] = (chd, pri)

        return chd

    def evaluate_node(self, node):
        if self.no_child(node.board):
            return 0
        eval_result = self.evaluation_func(node.board)
        if eval_result != 0:
            return eval_result
        if node.value is None:
            node.value, _ = self.get_calc(node.board)
        return node.value

    def run_mcts(self, board, simulations):
        root = MCTSNode(board)

        for _ in range(simulations):
            node = root
            search_path = [node]

            while node.children:
                node = self.select_child(node)
                search_path.append(node)

            if not self.is_terminal(node.board):
                self.expand_node(node)
            else:
                node.value = self.evaluate_node(node)

            value = self.evaluate_node(node)

            for node in reversed(search_path):
                node.visit_count += 1
                node.value_sum += value
                node.update_value()
                value = -value

        return root

    def get_best_move(self, board, simulations):
        if self.is_terminal(board):
            return None

        root = self.run_mcts(board, simulations)

        best_visit = -1
        best_move = None

        for move, (child, _) in root.children.items():
            if child is not None and child.visit_count > best_visit:
                best_visit = child.visit_count
                best_move = move

        return best_move

    def recommend_move(self, board_state, player, return_time=False):
        converted_board = self.convert_board_format(board_state, player)

        simulations = self._resolve_simulations(board_state)

        start_time = time.time()

        best_move = self.get_best_move(converted_board, simulations)

        elapsed_time = time.time() - start_time
        logger.info(
            "AI思考完成：耗时 %.2f 秒 | 设备: %s | 模拟次数: %d",
            elapsed_time,
            self.device.type,
            simulations,
        )

        if return_time:
            return (best_move, elapsed_time)
        return best_move

    def _resolve_simulations(self, board_state):
        if not self.adaptive_simulations:
            return self.base_simulations
        return get_adaptive_simulations(board_state)


