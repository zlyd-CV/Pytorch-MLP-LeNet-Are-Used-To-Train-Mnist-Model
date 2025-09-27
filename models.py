import torch.nn as nn


# 基于两个全连接层的多层感知机
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),  # pytorch创建层时参数权重默认使用Kaiming初始化，偏置为0
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        out = self.net(x)
        return out

# LeNet，它是最早发布的卷积神经网络之一
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(2, 2),  # 均值池化
            nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
            nn.AvgPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    # 选中行，快捷键：Tab or Shift Tab 可分别缩进或前向一个段落
    def forward(self, x):
        out = self.net(x)
        return out
