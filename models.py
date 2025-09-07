import torch.nn
import torch.nn as nn
from sympy.tensor.array.arrayop import Flatten


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
        self.fc=torch.nn.Linear(10,10)

    def forward(self, x):
        out = self.net(x)
        out=nn.Flatten()(out)
        out = self.fc(out)
        return out

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.net = nn.Sequential(
        # 卷积操作卷积层
        torch.nn.Conv2d(1,32,kernel_size=5,padding=2),
        # 归一化BN层
        torch.nn.BatchNorm2d(32),
        # 激活层 RELU函数
        torch.nn.ReLU(),
        # 最大池化
        torch.nn.MaxPool2d(2,2))
        self.fc=torch.nn.Linear(14*14*32,10)
    # 选中行，快捷键：Tab or Shift Tab 可分别缩进或前向一个段落
    def forward(self, x):
        out = self.net(x)
        out = nn.Flatten()(out)
        out=self.fc(out)
        return out

class SuperCNN(nn.Module):
    def __init__(self):
        super(SuperCNN, self).__init__()
        self.net = nn.Sequential(
            # 第1卷积块：32通道→64通道（通道数递增，增强特征表达）
            nn.Conv2d(1, 32, kernel_size=5, padding=2),  # 输入1通道（灰度图），输出32通道
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 池化后尺寸：28→14

            # 新增第2卷积块：64通道→128通道（加深网络，提取更细粒度特征）
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 小卷积核（3x3），减少参数且提升局部特征捕捉
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 池化后尺寸：14→7

            # 新增第3卷积块（可选，进一步提升性能）
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 池化后尺寸：7→3（最后一层池化后尺寸需能被整除）
        )
        # 调整全连接层输入维度：3（池化后尺寸）×3（尺寸）×128（通道数）= 1152
        self.fc = nn.Linear(3 * 3 * 128, 10)  # 输出10类（0-9）

    # 选中行，快捷键：Tab or Shift Tab 可分别缩进或前向一个段落
    def forward(self, x):
            out = self.net(x)
            out = nn.Flatten()(out)
            out = self.fc(out)
            return out




# 1. 通道注意力模块（SE Module）：让网络关注重要特征通道
class SEModule(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEModule, self).__init__()
        # 全局平均池化：将每个通道的特征压缩为 1 个值（通道的全局信息）
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        # 全连接层：压缩通道数（reduction 为压缩系数）→ 激活 → 恢复通道数
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),  # 压缩：通道数//16
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),  # 恢复：回到原通道数
            nn.Sigmoid()  # 输出每个通道的权重（0~1）
        )

    def forward(self, x):
        b, c, _, _ = x.size()  # b=批次大小, c=通道数
        # 全局平均池化：(b,c,h,w) → (b,c,1,1) → 展平为 (b,c)
        y = self.global_avg_pool(x).view(b, c)
        # 计算通道权重：(b,c) → (b,c)
        y = self.fc(y).view(b, c, 1, 1)
        # 权重乘回原特征：让重要通道的特征增强，不重要的减弱
        return x * y


# 2. 残差块（Residual Block）：解决深层网络梯度消失
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # 主路径：2个卷积层（3x3）+ BN + ReLU
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            SEModule(out_channels)  # 嵌入通道注意力
        )
        #  shortcut 路径：当输入输出通道/尺寸不匹配时，用1x1卷积调整
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        # 最终激活
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 残差连接：主路径输出 + shortcut 输出 → 激活
        out = self.main_path(x)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


# 3. 最终网络：基于残差块和注意力的深度CNN
class ResAttCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(ResAttCNN, self).__init__()
        # 初始卷积层：将1通道（灰度图）提升到64通道，尺寸28→28（padding=1）
        self.init_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 28→14
        )

        # 残差层：3个残差块，通道数逐步提升（64→128→256），尺寸逐步缩小
        self.residual_layers = nn.Sequential(
            # 残差块1：64→128， stride=2（14→7）
            ResidualBlock(in_channels=64, out_channels=128, stride=2),
            # 残差块2：128→128， stride=1（尺寸不变）
            ResidualBlock(in_channels=128, out_channels=128, stride=1),
            # 残差块3：128→256， stride=2（7→3，因7//2=3）
            ResidualBlock(in_channels=128, out_channels=256, stride=2),
            # 残差块4：256→256， stride=1（尺寸不变）
            ResidualBlock(in_channels=256, out_channels=256, stride=1)
        )

        # 全局平均池化（GAP）：将 (b,256,3,3) 压缩为 (b,256,1,1) → 展平为 (b,256)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # 分类头：Dropout + 全连接层（减少过拟合）
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),  # 随机丢弃50%神经元，抑制过拟合
            nn.Linear(256, num_classes)  # 256通道→10类（0-9）
        )

    def forward(self, x):
        # 前向传播流程
        x = self.init_conv(x)          # (b,1,28,28) → (b,64,14,14)
        x = self.residual_layers(x)    # (b,64,14,14) → (b,256,3,3)
        x = self.global_avg_pool(x)    # (b,256,3,3) → (b,256,1,1)
        x = x.view(x.size(0), -1)      # 展平：(b,256,1,1) → (b,256)
        x = self.classifier(x)         # 分类：(b,256) → (b,10)
        return x