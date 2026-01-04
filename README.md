# Alpha-USE

一个基于 AlphaZero 算法的强化学习 AI，专攻 **终极井字棋 (Ultimate Tic-Tac-Toe)**。

> **有趣的小知识**：本项目的核心代号 **USE** 代表 **US E**lection（美国大选）。这个命名的灵感来自于游戏机制：你需要先赢得局部的“州”（小棋盘），最终才能赢得“总统大选”（大棋盘）。

## 🎮 什么是终极井字棋？

终极井字棋 (Ultimate Tic-Tac-Toe) 是在一个巨大的 9x9 网格上进行的复杂策略游戏。

### 棋盘结构
*   **微观棋盘 (Micro-board)**: 一个标准的 3x3 井字棋盘。
*   **宏观棋盘 (Macro-board)**: 由 9 个微观棋盘组成的 3x3 大棋盘。

### 游戏规则
1.  **胜利条件**: 在宏观棋盘上连成一线（横、竖、斜）赢得 3 个小棋盘，即获得整局胜利。
2.  **核心约束 (限制机制)**: 你不能随心所欲地选择下在哪里！
    *   如果你在当前小棋盘的 **右上角** 格子落子...
    *   对手的下一步 **必须** 下在宏观大棋盘的 **右上角** 那个小棋盘里。
3.  **自由移动 (Free Move)**: 如果你被指派前往的小棋盘已经决出胜负（被占领）或填满（平局），你可以在全盘任意一个 **未终结** 的小棋盘内落子。

这种“约束”机制创造了极深的战略层次，你必须为了控制对手的下一步动向而牺牲局部的利益。

## 🧠 技术细节

本项目基于 [alpha-zero-general](https://github.com/suragnair/alpha-zero-general) 框架，但针对终极井字棋的高复杂度进行了深度定制。

### 架构亮点
不同于传统的简单棋盘表示，**Alpha-USE** 采用了一套先进的 **6通道输入状态 (6-Channel Input State)**，帮助神经网络“看懂”棋局结构：

*   **Channel 0**: 我方棋子位置 (微观)
*   **Channel 1**: 敌方棋子位置 (微观)
*   **Channel 2**: 我方已占领的小棋盘 (宏观胜利状态)
*   **Channel 3**: 敌方已占领的小棋盘 (宏观胜利状态)
*   **Channel 4**: 平局的小棋盘 (宏观平局状态)
*   **Channel 5**: **合法落子掩码 (Legal Move Mask)** (处理约束逻辑)

这种 `9x9x6` 的张量结构让卷积神经网络 (CNN) 能够高效地同时处理战术细节（局部）和战略大局（全局）。

### 当前棋力
*   **状态**: 在 **RTX 4060 (Laptop)** 上训练约 **24 小时**。
*   **水平**: **初具人形**。

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/YourUsername/alpha-use.git
cd alpha-use
pip install -r requirements.txt
```

### 使用方法

与 AI 对战：

```bash
# 启动游戏：人类 vs AI
python GUI.py
```

训练模型：

```bash
# 修改 main.py 中的 args 配置以加载最新的 checkpoint
python main.py
```

## 🤝 致谢

*   感谢 [suragnair/alpha-zero-general](https://github.com/suragnair/alpha-zero-general) 提供的出色的 AlphaZero PyTorch 实现。
*   致敬 DeepMind 的 AlphaGo Zero 原始论文。