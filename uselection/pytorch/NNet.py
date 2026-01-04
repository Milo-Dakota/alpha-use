import os
import sys
import time

import numpy as np
from tqdm import tqdm

sys.path.append('../../')
from utils import *
from NeuralNet import NeuralNet

import torch
import torch.optim as optim

from .USENNet import USENNet as onnet

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 512,
})

class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = onnet(game, args)
        self.action_size = game.getActionSize()

        if args.cuda:
            self.nnet.cuda()

    def convert_to_six_planes(self, board):
        """
        核心数据转换工厂
        输入: 10x9 numpy array (MCTS Logic Board)
        输出: 6x9x9 numpy array (Neural Net Input)
        """
        # 1. 拆解数据
        raw_board = board[:9, :]
        last_move = int(board[9, 0])

        # --- Channel 0 & 1: 棋子位置 ---
        # 假设当前玩家视角的棋子是 1
        p1_pieces = (raw_board == 1).astype(np.float32)
        p2_pieces = (raw_board == -1).astype(np.float32)

        # --- 计算宏观状态 (Channel 2, 3, 4) ---
        macro_self = np.zeros((9, 9), dtype=np.float32)
        macro_opp = np.zeros((9, 9), dtype=np.float32)
        macro_draw = np.zeros((9, 9), dtype=np.float32)
        
        # 记录每个小棋盘的状态: 0=Active, 1=Terminated (Won or Draw)
        micro_board_status = [0] * 9 

        for r in range(3):
            for c in range(3):
                # 提取 3x3 小棋盘
                sub = raw_board[r*3:(r+1)*3, c*3:(c+1)*3]
                
                winner = 0 # 0:None, 1:Self, -1:Opp, 2:Draw
                
                # 快速胜负判定 (Numpy)
                # 行列检查
                row_sum = np.sum(sub, axis=1)
                col_sum = np.sum(sub, axis=0)
                if np.any(row_sum == 3) or np.any(col_sum == 3): winner = 1
                elif np.any(row_sum == -3) or np.any(col_sum == -3): winner = -1
                else:
                    # 对角线检查
                    d1 = sub.trace()
                    d2 = np.fliplr(sub).trace()
                    if d1 == 3 or d2 == 3: winner = 1
                    elif d1 == -3 or d2 == -3: winner = -1
                    elif not np.any(sub == 0): winner = 2 # 平局

                # 填充 Channel 2/3/4
                if winner != 0:
                    micro_board_status[r*3+c] = 1
                    target_slice = (slice(r*3, (r+1)*3), slice(c*3, (c+1)*3))
                    if winner == 1:
                        macro_self[target_slice] = 1
                    elif winner == -1:
                        macro_opp[target_slice] = 1
                    elif winner == 2:
                        macro_draw[target_slice] = 1

        # --- Channel 5: Legal Move Mask ---
        legal_mask = np.zeros((9, 9), dtype=np.float32)
        
        # 计算下一个合法落子的小棋盘索引
        target_micro_idx = -1
        if last_move != -1:
            prev_r, prev_c = last_move // 9, last_move % 9
            target_micro_idx = (prev_r % 3) * 3 + (prev_c % 3)
        
        # 判断是否 Free Move (开局 或 目标小棋盘已终结)
        is_free_move = (target_micro_idx == -1) or (micro_board_status[target_micro_idx] == 1)
        
        if is_free_move:
            # 所有 active 的小棋盘的空格都可以下
            for idx in range(9):
                if micro_board_status[idx] == 0:
                    r_base, c_base = (idx // 3) * 3, (idx % 3) * 3
                    sub_empty = (raw_board[r_base:r_base+3, c_base:c_base+3] == 0)
                    legal_mask[r_base:r_base+3, c_base:c_base+3] = sub_empty
        else:
            # 只能在目标小棋盘下
            r_base, c_base = (target_micro_idx // 3) * 3, (target_micro_idx % 3) * 3
            sub_empty = (raw_board[r_base:r_base+3, c_base:c_base+3] == 0)
            legal_mask[r_base:r_base+3, c_base:c_base+3] = sub_empty

        return np.stack([p1_pieces, p2_pieces, macro_self, macro_opp, macro_draw, legal_mask])

    def train(self, examples):
        """
        examples: list of (board_10x9, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters(), lr=args.lr)

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / args.batch_size)

            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                
                # 提取数据 (boards 是 10x9 的)
                boards_raw, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                
                # --- 关键步骤：转换数据 ---
                boards_6ch = [self.convert_to_six_planes(b) for b in boards_raw]
                boards = torch.FloatTensor(np.array(boards_6ch).astype(np.float64))
                
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                if args.cuda:
                    boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                # compute output
                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                # SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def predict(self, board):
        """
        board: 10x9 numpy array
        """
        # 1. 转换数据 (10x9 -> 6x9x9)
        board_6ch = self.convert_to_six_planes(board)
        
        # 2. 转 Tensor 并增加 Batch 维度 -> (1, 6, 9, 9)
        board_tensor = torch.FloatTensor(board_6ch.astype(np.float64)).unsqueeze(0)
        
        if args.cuda: 
            board_tensor = board_tensor.contiguous().cuda()
        
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board_tensor)

        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])