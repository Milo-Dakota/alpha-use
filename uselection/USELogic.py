import numpy as np

class Board():
    def __init__(self, n=9):
        "Set up initial board configuration."
        self.n = n
        
        # 10x9 矩阵
        # [0-8][:]: 实际棋盘 (1=Player1, -1=Player2, 0=Empty)
        # [9][0]:   上一手的位置 (0-80), -1表示无限制
        self.pieces = np.zeros((self.n + 1, self.n))
        self.pieces[9][0] = -1 # 初始状态无限制

    # add [][] indexer syntax to the Board
    def __getitem__(self, index): 
        return self.pieces[index]

    def get_legal_moves(self):
        """Returns all the legal moves for the given color.
        (1 for white, -1 for black)
        """
        moves = set()
        
        last_move = int(self.pieces[9][0])
        
        # 1. 如果是开局，所有空位都合法
        if last_move == -1:
            return self._get_all_empty_moves()

        # 2. 计算上一手指向的 3x3 区域 (Constraint)
        # last_move 是 0-80 的整数
        prev_row = last_move // 9
        prev_col = last_move % 9
        
        # 下一步必须落在 (row%3, col%3) 这个大格子里
        constraint_r = prev_row % 3
        constraint_c = prev_col % 3
        
        # 3. 检查该区域是否可用
        if self._is_macro_playable(constraint_r, constraint_c):
            # 只能在该区域下
            base_r, base_c = constraint_r * 3, constraint_c * 3
            for r in range(3):
                for c in range(3):
                    if self.pieces[base_r + r][base_c + c] == 0:
                        moves.add((base_r + r, base_c + c))
        else:
            # Free Move: 全局任意空位
            return self._get_all_empty_moves()
            
        return list(moves)

    def has_legal_moves(self):
        # 只要有空位就有合法移动 (除非平局)
        return len(self.get_legal_moves) > 0

    def execute_move(self, move, color):
        """Perform the given move on the board."""
        # move is (x, y) tuple
        x, y = move
        # 1. 落子
        self.pieces[x][y] = color
        
        # 2. 更新约束 (记录这一手的位置 0-80)
        action_id = x * 9 + y
        self.pieces[9][0] = action_id

    def check_win(self):
        """
        检测游戏是否结束。
        Return: 1(赢), -1(输), 0(未结束), 2(平局)
        注意：这里的赢家是相对于棋子颜色而言的，不是相对于 'current player'
        """
        # 1. 构建 Macro Board (3x3)
        macro_board = np.zeros((3, 3))
        for r in range(3):
            for c in range(3):
                status = self._check_micro_win(r, c)
                macro_board[r][c] = status # 1, -1, 0, or 0.1(draw/full)
        
        # 2. 检查 Macro Board 是否连线
        winner = self._check_3x3_win(macro_board)
        if winner != 0:
            return winner
            
        # 3. 检查平局 (全盘满了)
        # 只要 9x9 区域里没有 0 了，就是平局
        if not np.any(macro_board == 0):
            return 0.1 # Draw
            
        return 0

    # --- Helper Functions ---

    def _get_all_empty_moves(self):
        moves = set()
        for x in range(3):
            for y in range(3):
                if self._is_macro_playable(x, y):
                    base_r, base_c = x * 3, y * 3
                    for r in range(3):
                        for c in range(3):
                            if self.pieces[base_r + r][base_c + c] == 0:
                                moves.add((base_r + r, base_c + c))
        return list(moves)

    def _is_macro_playable(self, r, c):
        # 检查大格子 (r, c) 是否还能下
        # 如果已经有人赢了(1/-1) 或者 满了(2)，就不能指定下这里 -> 触发 Free Move
        status = self._check_micro_win(r, c)
        return status == 0 # 0 表示还在进行中

    def _check_micro_win(self, macro_r, macro_c):
        # 检查具体的 3x3 小格子谁赢了
        base_r, base_c = macro_r * 3, macro_c * 3
        # 提取 3x3 切片
        sub_board = self.pieces[base_r : base_r+3, base_c : base_c+3]
        
        # 检查连线
        winner = self._check_3x3_win(sub_board)
        if winner != 0:
            return winner
        
        # 检查是否满了 (Draw)
        if not np.any(sub_board == 0):
            return 0.1 # Draw
        
        return 0

    def _check_3x3_win(self, board_3x3):
        # 检查任意 3x3 矩阵 (micro 或 macro) 是否有连线
        # 行
        for i in range(3):
            if abs(sum(board_3x3[i, :])) == 3: return np.sign(sum(board_3x3[i, :]))
        # 列
        for i in range(3):
            if abs(sum(board_3x3[:, i])) == 3: return np.sign(sum(board_3x3[:, i]))
        # 对角线
        if abs(board_3x3[0,0] + board_3x3[1,1] + board_3x3[2,2]) == 3: 
            return np.sign(board_3x3[1,1])
        if abs(board_3x3[0,2] + board_3x3[1,1] + board_3x3[2,0]) == 3: 
            return np.sign(board_3x3[1,1])
            
        return 0