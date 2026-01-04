import pygame
import sys
import numpy as np
import time

from uselection.USEGame import USEGame

# --- 配置常量 ---
BOARD_SIZE = 9
SQUARE_SIZE = 60  # 每个格子的像素大小
MARGIN = 2        # 格子之间的间隔
width = BOARD_SIZE * (SQUARE_SIZE + MARGIN) + MARGIN
height = BOARD_SIZE * (SQUARE_SIZE + MARGIN) + MARGIN + 50 # 底部留出状态栏

# 颜色定义 (R, G, B)
COLOR_BG = (40, 40, 40)       # 背景色
COLOR_GRID = (200, 200, 200)  # 空格子颜色
COLOR_P1 = (0, 0, 0)          # 玩家 1 (黑/O)
COLOR_P2 = (240, 240, 240)    # 玩家 -1 (白/X)
COLOR_VALID = (100, 200, 100) # 有效移动提示色 (可选)
COLOR_TEXT = (255, 255, 255)

class GameUI:
    def __init__(self, game, ai_player=None, ai_turn=-1):
        """
        :param game: USEGame 实例
        :param ai_player: 一个函数，接受 board 并返回 action (int)。如果为 None，则是双人对战。
        :param ai_turn: AI 执什么棋？1 为先手，-1 为后手。
        """
        self.game = game
        self.board = game.getInitBoard()
        self.cur_player = 1  # 1 Starts
        self.game_over = False
        self.winner = 0
        
        # AI 配置
        self.ai_player = ai_player
        self.ai_turn = ai_turn

        # Pygame 初始化
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("USEGame Arena (9x9)")
        self.font = pygame.font.SysFont('Arial', 24)

    def draw_board(self):
        self.screen.fill(COLOR_BG)
        
        # 获取当前合法的移动，用于高亮显示（可选）
        valid_moves = self.game.getValidMoves(self.board)
        
        # 注意：虽然 board 可能是 (10, 9)，我们只画前 9x9
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                x = MARGIN + c * (SQUARE_SIZE + MARGIN)
                y = MARGIN + r * (SQUARE_SIZE + MARGIN)
                
                # 确定格子颜色
                piece = self.board[r][c]
                color = COLOR_GRID
                
                if piece == 1:
                    color = COLOR_P1
                elif piece == -1:
                    color = COLOR_P2
                
                # 绘制方块
                pygame.draw.rect(self.screen, color, (x, y, SQUARE_SIZE, SQUARE_SIZE))
                
                # 如果是当前人类玩家的回合，且该格子为空且合法，画一个小圆点提示
                if not self.game_over and piece == 0 and valid_moves[r*BOARD_SIZE + c] and \
                   (self.ai_player is None or self.cur_player != self.ai_turn):
                    center = (x + SQUARE_SIZE//2, y + SQUARE_SIZE//2)
                    pygame.draw.circle(self.screen, (150, 150, 150), center, 5)

    def draw_status(self):
        text_str = ""
        if self.game_over:
            if self.winner == 1:
                text_str = "Game Over: Player 1 (Black) Wins!"
            elif self.winner == -1:
                text_str = "Game Over: Player 2 (White) Wins!"
            else:
                text_str = "Game Over: Draw!"
        else:
            p_name = "Player 1 (Black)" if self.cur_player == 1 else "Player 2 (White)"
            if self.ai_player and self.cur_player == self.ai_turn:
                p_name += " [AI Thinking...]"
            else:
                p_name += " [Your Turn]"
            text_str = f"Turn: {p_name}"

        text_surface = self.font.render(text_str, True, COLOR_TEXT)
        self.screen.blit(text_surface, (10, height - 40))

    def handle_click(self, pos):
        if self.game_over:
            return

        # 如果是 AI 的回合，忽略点击
        if self.ai_player and self.cur_player == self.ai_turn:
            return

        x, y = pos
        # 排除点击到底部状态栏的情况
        if y > width: 
            return

        col = x // (SQUARE_SIZE + MARGIN)
        row = y // (SQUARE_SIZE + MARGIN)

        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
            action = row * BOARD_SIZE + col
            
            # 检查合法性
            valids = self.game.getValidMoves(self.board)
            if valids[action]:
                self.step(action)
            else:
                print(f"Invalid move: {row}, {col}")

    def step(self, action):
        # 执行移动
        self.board, self.cur_player = self.game.getNextState(self.board, self.cur_player, action)
        
        # 检查游戏结束
        # USEGame rules: 1=P1 win, -1=P2 win, 0.1=Draw/End, 0=Not ended
        res = self.game.getGameEnded(self.board, self.cur_player) * self.cur_player
        
        if res != 0:
            self.game_over = True
            self.winner = res
            print(f"Game End Code: {res}")

    def run(self):
        clock = pygame.time.Clock()
        running = True

        while running:
            # 1. 处理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(event.pos)

            # 2. 绘制
            self.draw_board()
            self.draw_status()
            pygame.display.flip()

            # 3. AI 逻辑
            if not self.game_over and self.ai_player and self.cur_player == self.ai_turn:
                # 强制刷新一下界面，否则 AI 计算时界面会卡在上一帧
                pygame.event.pump() 
                
                # 简单的延迟，让人类看清 AI 下棋的节奏
                time.sleep(0.1) 
                
                # 获取 Canonical Board 供 AI 使用
                canonical_board = self.game.getCanonicalForm(self.board, self.cur_player)
                
                # 计算动作
                action = self.ai_player(canonical_board)
                
                # 执行动作
                print(f"AI chooses action: {action // 9}, {action % 9}")
                self.step(action)

            clock.tick(30)

        pygame.quit()

# --- 主程序入口 ---
if __name__ == "__main__":
    # 1. 初始化游戏
    g = USEGame()

    try:
        from MCTS import MCTS
        # 假设你的 NNet 在某个路径，比如 othello.NNet
        # 这里需要你根据实际项目结构修改 import
        from uselection.pytorch.NNet import NNetWrapper as NNet 

        # 模拟加载参数
        class DotDict(dict):
            def __getattr__(self, name): return self[name]
        
        args = DotDict({'numMCTSSims': 200, 'cpuct': 1.0})
        
        # 实例化
        n1 = NNet(g)
        n1.load_checkpoint('./temp/', 'best.pth.tar') # 加载你的模型
        
        mcts = MCTS(g, n1, args)

        # 定义 AI 动作选择函数
        # temp=0 表示确定性选择（最强），temp=1 表示带探索
        ai_brain = lambda x: np.argmax(mcts.getActionProb(x, temp=0))

        # 在ai_turn=中配置你希望ai执先手还是后手，1表示先手，-1时表示后手
        print("Starting Game: Human (Black) vs AI (White)")
        ui = GameUI(g, ai_player=ai_brain, ai_turn=1)
        ui.run()

    except ImportError as e:
        print(f"Could not load AI modules: {e}")
        print("Falling back to Human vs Human mode.")
        ui = GameUI(g)
        ui.run()