from gomoku_ai_core import GomokuAI

DEFAULT_CPU_SIMULATIONS = 200


class GomokuCPUAI:
    """面向CPU环境的轻量封装。"""

    def __init__(self, model_path='run10_2000.pth', simulations=DEFAULT_CPU_SIMULATIONS, c_puct=1.0, use_rand=0.01):
        self.model_path = model_path
        self.simulations = simulations
        self.c_puct = c_puct
        self.use_rand = use_rand
        self.ai = GomokuAI(
            model_path=model_path,
            device='cpu',
            simulations=simulations,
            c_puct=c_puct,
            use_rand=use_rand,
            adaptive_simulations=False,
        )

    def recommend_move(self, board_state, player, return_time=False):
        return self.ai.recommend_move(board_state, player, return_time=return_time)


def create_gomoku_ai(model_path='run10_2000.pth', simulations=DEFAULT_CPU_SIMULATIONS):
    return GomokuCPUAI(model_path, simulations=simulations)


def get_ai_move(board_state, player, model_path='run10_2000.pth', simulations=DEFAULT_CPU_SIMULATIONS):
    ai = GomokuCPUAI(model_path, simulations=simulations)
    return ai.recommend_move(board_state, player, return_time=False)


if __name__ == "__main__":
    test_board = [[0] * 15 for _ in range(15)]
    move = get_ai_move(test_board, 1)
    print(f"推荐落子位置: {move}")