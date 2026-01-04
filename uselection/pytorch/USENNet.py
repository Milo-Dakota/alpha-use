import sys
sys.path.append('..')
from utils import *

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class USENNet(nn.Module):
    """
    USE (US Election / Ultimate Tic-Tac-Toe) Neural Network
    基于 9x9x6 的多通道输入架构
    """
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = (9,9)
        self.action_size = game.getActionSize()          # 应该是 81
        self.args = args

        super(USENNet, self).__init__()

        # --- 定义输入通道数 ---
        # 1. P1 Pieces (我方棋子)
        # 2. P2 Pieces (敌方棋子)
        # 3. Self Macro-Win (我方赢下的州/小棋盘)
        # 4. Opp Macro-Win (敌方赢下的州/小棋盘)
        # 5. Micro-Draw (陷入僵局/平局的州)
        # 6. Legal Moves Mask (当前的合法竞选/落子区域)
        self.input_channels = 6 

        # --- 卷积层 ---
        # Conv1: 接收 6 层特征图 -> 输出 num_channels (例如 64 或 512)
        self.conv1 = nn.Conv2d(self.input_channels, args.num_channels, 3, stride=1, padding=1)
        # Conv2-4: 保持特征提取，padding=1 保证尺寸维持在 9x9
        self.conv2 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)

        # --- 批归一化层 (Batch Normalization) ---
        self.bn1 = nn.BatchNorm2d(args.num_channels)
        self.bn2 = nn.BatchNorm2d(args.num_channels)
        self.bn3 = nn.BatchNorm2d(args.num_channels)
        self.bn4 = nn.BatchNorm2d(args.num_channels)

        # --- 全连接层 ---
        # 输入维度: num_channels * 9 * 9
        self.fc1 = nn.Linear(args.num_channels * self.board_x * self.board_y, 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        # Policy Head: 输出 81 个动作的概率 logits
        self.fc3 = nn.Linear(512, self.action_size)

        # Value Head: 输出当前局面的胜率评估 (-1 到 1)
        self.fc4 = nn.Linear(512, 1)

    def forward(self, s):
        # s: 输入数据，形状通常是 (batch_size, 486) 或者已经 reshape 好的
        # 我们将其强制 reshape 为 (batch_size, 6, 9, 9) 以匹配我们的设计
        # s shape: batch_size x 6 x 9 x 9
        s = s.view(-1, self.input_channels, self.board_x, self.board_y)
        
        # 卷积流程 (尺寸全程保持 9x9)
        s = F.relu(self.bn1(self.conv1(s)))                          # batch_size x num_channels x 9 x 9
        s = F.relu(self.bn2(self.conv2(s)))                          # batch_size x num_channels x 9 x 9
        s = F.relu(self.bn3(self.conv3(s)))                          # batch_size x num_channels x 9 x 9
        s = F.relu(self.bn4(self.conv4(s)))                          # batch_size x num_channels x 9 x 9
        
        # 展平 (Flatten)
        s = s.view(-1, self.args.num_channels * self.board_x * self.board_y)

        # 全连接流程
        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)  # batch_size x 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)  # batch_size x 512

        # 输出层
        pi = self.fc3(s)  # Policy
        v = self.fc4(s)   # Value

        return F.log_softmax(pi, dim=1), torch.tanh(v)