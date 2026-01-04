from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .USELogic import Board
import numpy as np

class USEGame(Game):
    square_content = {
        -1: "X",
        +0: "-",
        +1: "O"
    }

    @staticmethod
    def getSquarePiece(piece):
        return USEGame.square_content[piece]

    def __init__(self):
        self.n = 9

    def getInitBoard(self):
        # 初始状态：返回 10x9 的内部逻辑表示
        # MCTS 和 Logic 依然基于这个轻量级表示跑
        b = Board(self.n)
        return np.array(b.pieces)

    def getBoardSize(self):
        # 注意：这里返回的是神经网络输入的大小！
        # 即使我们内部用 10x9，但告诉 NNet 我们是 (6, 9, 9)
        return (6, 9, 9)

    def getActionSize(self):
        return self.n * self.n # 81

    def getNextState(self, board, player, action):
        # 输入 board 是 10x9 的
        b = Board(self.n)
        b.pieces = np.copy(board)
        move = (int(action/self.n), action%self.n)
        b.execute_move(move, player)
        return (b.pieces, -player)

    def getValidMoves(self, board):
        # 输入 board 是 10x9 的
        valids = [0]*self.getActionSize()
        b = Board(self.n)
        b.pieces = np.copy(board)
        legalMoves = b.get_legal_moves()
        for x, y in legalMoves:
            valids[self.n*x+y]=1
        return np.array(valids)

    def getGameEnded(self, board, player):
        # 输入 board 是 10x9 的
        b = Board(self.n)
        b.pieces = np.copy(board)
        winner = b.check_win()
        if winner == player:
            return 1
        elif winner == -player:
            return -1
        elif winner == 0.1: # Draw
            return 1e-4 # small non-zero value for draw
        else:
            return 0

    def getCanonicalForm(self, board, player):
        # 返回 10x9，只处理视角 (1/-1)
        canonical_board = np.copy(board)
        last_move = canonical_board[9][0]
        canonical_board[9][0] = 0
        canonical_board = canonical_board * player
        canonical_board[9][0] = last_move
        return canonical_board # 保持 10x9 给 MCTS 用

    def getSymmetries(self, board, pi):
        # board: 10x9 numpy array (前9行是棋盘, board[9][0]是 last_move)
        # pi: 长度 81 的策略向量
        
        # 1. 拆分数据
        raw_board = board[:9, :]
        last_move_idx = int(board[9, 0])
        
        # 将策略向量重塑为 9x9 以便旋转
        pi_board = np.reshape(pi, (self.n, self.n))
        l = []

        # --- 辅助函数：计算坐标旋转 (0-80 index) ---
        def rotate_idx(idx, k):
            if idx == -1: return -1
            r, c = idx // 9, idx % 9
            for _ in range(k):
                # 逆时针旋转 90 度: (r, c) -> (N-1-c, r)
                # 这里 N=9, 所以是 (8-c, r)
                r, c = 8 - c, r
            return r * 9 + c

        # --- 辅助函数：计算坐标镜像 (0-80 index) ---
        def flip_idx(idx):
            if idx == -1: return -1
            r, c = idx // 9, idx % 9
            # 左右翻转 (fliplr): (r, c) -> (r, N-1-c)
            c = 8 - c
            return r * 9 + c

        # 2. 生成所有旋转/镜像组合
        for i in range(1, 5): # 旋转 0(4), 1, 2, 3 次
            for j in [True, False]: # 是否镜像
                
                # A. 旋转棋盘内容 (图像旋转)
                newB_raw = np.rot90(raw_board, i)
                newPi = np.rot90(pi_board, i)
                
                # B. 旋转约束坐标 (数学变换)
                new_last_move = rotate_idx(last_move_idx, i)
                
                # C. 如果需要镜像
                if j:
                    newB_raw = np.fliplr(newB_raw)
                    newPi = np.fliplr(newPi)
                    new_last_move = flip_idx(new_last_move)
                
                # D. 重新组装为 10x9 格式
                newB = np.zeros_like(board)
                newB[:9, :] = newB_raw
                newB[9, 0] = new_last_move # 更新变换后的约束位置
                
                l += [(newB, list(newPi.ravel()))]
        
        return l

    def stringRepresentation(self, board):
        # board 是 6x9x9 的 float array
        return board.tostring()