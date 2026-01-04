import numpy as np

class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # 随机生成一个动作
        a = np.random.randint(self.game.getActionSize())
        # 获取当前合法的动作列表
        valids = self.game.getValidMoves(board)
        # 如果随机到的动作不合法，就一直重随
        while valids[a] != 1:
            a = np.random.randint(self.game.getActionSize())
        return a

class HumanUSEPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # 获取合法动作
        valid = self.game.getValidMoves(board)
        
        # 打印提示
        print("合法落子点 (Row, Col):")
        for i in range(len(valid)):
            if valid[i]:
                # 把 0-80 转换成坐标打印出来，方便你看
                print(f"({int(i/9)}, {int(i%9)})", end=" ")
        print("") # 换行

        while True:
            # 等待用户输入
            input_move = input("请输入落子坐标 (格式: 行 列，例如: 0 5): ")
            input_a = input_move.split(" ")
            
            if len(input_a) == 2:
                try:
                    x, y = [int(i) for i in input_a]
                    # 检查坐标范围 (0-8)
                    if ((0 <= x < 9) and (0 <= y < 9)):
                        a = 9 * x + y
                        # 检查是否符合规则（比如是否在限制的框内）
                        if valid[a]:
                            return a
                        else:
                            print("虽然坐标在棋盘内，但根据规则这一步不合法（受上一手限制）。")
                    else:
                        print("坐标超出范围 (0-8)。")
                except ValueError:
                    print("请输入整数。")
            else:
                print("格式错误，请输入两个数字，用空格隔开。")