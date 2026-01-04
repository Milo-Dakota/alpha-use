from Arena import Arena
from MCTS import MCTS
# --- 修改 1: 导入你的类 ---
from uselection.USEGame import USEGame
from uselection.USEPlayers import RandomPlayer, HumanUSEPlayer
from uselection.pytorch.NNet import NNetWrapper as NNet

import numpy as np
from utils import *

"""
这是一个测试脚本。
用途：
1. 验证你的游戏规则 (Game/Logic) 是否写对了 (Human vs Random)
2. 验证你的神经网络架构 (NNet) 是否能跑通 (Human vs Untrained AI)
"""

g = USEGame()

# --- 定义各种玩家 ---

# 1. 随机玩家 (用来测试规则有没有 bug)
rp = RandomPlayer(g).play

# 2. 人类玩家 (你自己)
hp = HumanUSEPlayer(g).play

# 3. 神经网络玩家 (即便没训练，也可以初始化一个随机权重的网络来测试代码跑不跑得通)
# 如果你想测试 NNet 结构是否报错，把下面的注释打开
n1 = NNet(g)

args = dotdict({'numMCTSSims': 100, 'cpuct': 1.0})

# 实例化
n1 = NNet(g)
n1.load_checkpoint('./temp/', 'best.pth.tar') # 加载你的模型

mcts = MCTS(g, n1, args)

# 定义 AI 动作选择函数
# temp=0 表示确定性选择（最强），temp=1 表示带探索
ai_brain = lambda x: np.argmax(mcts.getActionProb(x, temp=0))

# --- 设置对战双方 ---

print("请选择对战模式:")
print("1. 人类 vs 随机 (推荐: 先用这个测规则)")
print("2. 随机 vs 随机 (推荐: 快速跑几局看有没有报错)")
# print("3. 人类 vs 未训练的AI (测试 NNet)")

mode = input("请输入数字 (1/2): ")

if mode == '1':
    player1 = rp
    player2 = ai_brain
elif mode == '2':
    player1 = rp
    player2 = rp
else:
    print("默认: 人类 vs 随机")
    player1 = hp
    player2 = rp

# --- 开始竞技场 ---
# display=USEGame.display 会调用你在 Game 里写的打印函数
arena = Arena(player1, player2, g)

# playGames(2) 表示对战 2 局，verbose=True 表示打印每一步棋盘
print(arena.playGames(40, verbose=False))