import logging
import coloredlogs

from Coach import Coach
# --- 修改 1: 导入你的 Game 和 NNet ---
from uselection.USEGame import USEGame as Game
from uselection.pytorch.NNet import NNetWrapper as nn
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 1000,           # 总共训练多少轮
    'numEps': 100,               # 每一轮自我对弈多少局 (建议：调试时设为 10，正式跑设为 50-100)
    'tempThreshold': 15,        # 前 15 步按概率落子，后面按最大概率落子
    'updateThreshold': 0.6,     # 新模型胜率超过 60% 才能取代旧模型
    'maxlenOfQueue': 200000,    # 训练数据池的大小
    'numMCTSSims': 100,         # 每一步 MCTS 树搜索次数
    'arenaCompare': 40,         # 竞技场对战次数 (新旧模型各执先手 20 局)
    'cpuct': 1,                 # MCTS 探索系数

    'checkpoint': './temp/',    # 模型保存路径
    'load_model': True,        # 是否加载之前的模型继续训练
    'load_folder_file': ('./temp/','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
})


def main():
    log.info('Loading %s...', Game.__name__)
    
    # --- 修改 3: 初始化不需要参数 ---
    # Othello 需要 Game(6)，但 UTTT 是固定的，我们在类里写死了 self.n = 9
    g = Game() 

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()