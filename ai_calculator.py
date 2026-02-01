from gomoku_ai_core import GomokuAI


class AICalculator:
    """主程序调用的AI计算器封装。"""

    def __init__(
        self,
        model_path='run10_2000.pth',
        adaptive_simulations=True,
        use_mixed_precision=None,
    ):
        self.model_path = model_path
        self.adaptive_simulations = adaptive_simulations
        self.use_mixed_precision = use_mixed_precision
        self.ai = None

    def _fallback_move(self, board_state):
        if board_state[7][7] == 0:
            return ((7, 7), 0)

        for i in range(15):
            for j in range(15):
                if board_state[i][j] == 0:
                    return ((i, j), 0)
        return (None, 0)

    def _ensure_ai(self):
        if self.ai is None:
            self.ai = GomokuAI(
                self.model_path,
                adaptive_simulations=self.adaptive_simulations,
                use_mixed_precision=self.use_mixed_precision,
            )

    def get_best_move(self, board_state, player):
        """
        获取最佳落子位置。

        Returns:
            ((row, col), elapsed_time)
        """
        try:
            self._ensure_ai()
        except Exception:
            return self._fallback_move(board_state)

        try:
            result = self.ai.recommend_move(board_state, player, return_time=True)
            move, elapsed_time = result
            if move is None:
                return (None, elapsed_time)
            return (move, elapsed_time)
        except Exception:
            # 如果计算失败，清理实例以便下次重建
            self.ai = None
            return self._fallback_move(board_state)


if __name__ == "__main__":
    test_board = [[0] * 15 for _ in range(15)]
    calculator = AICalculator()
    move = calculator.get_best_move(test_board, 2)
    print(f"推荐落子: {move}")