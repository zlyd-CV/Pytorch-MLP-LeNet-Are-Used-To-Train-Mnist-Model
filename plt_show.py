import torch
import matplotlib.pyplot as plt
import numpy as np
import time  # 用于模拟训练耗时

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei"]
# 解决负号显示问题（可选）
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号


def show_mnist_images(images, labels, num_show=5, fig_title="MNIST Sample Images"):
    """
    展示 MNIST 数据集的图像及对应标签

    参数说明：
    - images: 图像数据，格式为 PyTorch Tensor，形状需为 [batch_size, 1, height, width]（单通道）
    - labels: 图像对应的标签，格式为 PyTorch Tensor，形状为 [batch_size]
    - num_show: 要展示的图像数量，默认展示前5张（需小于等于 batch_size）
    - fig_title: 整个画布的标题，默认值为 "MNIST Sample Images"
    """
    # 1. 校验输入合法性（避免因参数错误导致报错）
    if not isinstance(images, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise TypeError("images 和 labels 必须是 PyTorch Tensor 类型")
    if images.shape[0] < num_show:
        raise ValueError(f"图像批次大小（{images.shape[0]}）小于要展示的数量（{num_show}），请减少 num_show")
    if images.shape[1] != 1:
        raise ValueError("仅支持单通道（MNIST 类）图像，images 第二维度需为 1")

    # 2. 计算子图布局（默认按“1行N列”展示，若数量多可自动调整为“2行N列”等，更美观）
    rows = 1 if num_show <= 5 else 2  # 数量≤5用1行，＞5用2行
    cols = num_show if rows == 1 else (num_show + 1) // 2  # 按行数分配列数，避免空图

    # 3. 创建画布（根据布局调整大小，确保图像不拥挤）
    fig_size_w = cols * 2  # 每列占2个单位宽度
    fig_size_h = rows * 2  # 每行占2个单位高度
    plt.figure(figsize=(fig_size_w, fig_size_h))
    plt.suptitle(fig_title, fontsize=12, y=1.00)  # 画布标题，轻微上移避免遮挡子图

    # 4. 循环展示每张图像
    for i in range(num_show):
        # 处理图像格式：Tensor([1,28,28]) → 去掉通道维度 → 转为numpy数组
        img = images[i].squeeze().numpy()
        # 处理标签：Tensor 单元素 → Python 数字
        label = labels[i].item()

        # 创建子图（rows行cols列，第i+1个位置）
        ax = plt.subplot(rows, cols, i + 1)
        # 显示灰度图（cmap='gray'确保灰度配色，vmin/vmax固定像素范围0-1，对比更清晰）
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        # 隐藏坐标轴刻度（避免干扰图像展示）
        ax.set_xticks([])
        ax.set_yticks([])
        # 在图像下方标注标签（字体稍大，便于查看）
        ax.set_xlabel(f"Label: {label}", fontsize=10)

    # 调整子图间距（自动适配，避免标题/标签重叠）
    plt.tight_layout()
    # 显示图像（运行后弹出窗口，关闭窗口后程序继续执行）
    plt.show()


def evaluation_indicators(epochs):
    """初始化动态图表"""
    # 设置中文字体
    plt.rcParams["font.family"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # 启用交互模式
    plt.ion()

    # 创建图表和轴对象
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("training_target")
    ax.set_xlabel("epoch")
    ax.set_ylabel("acc/loss")
    ax.set_xlim(1, epochs)  # x轴范围固定为总迭代次数
    ax.set_ylim(0, 1.1)  # y轴范围根据实际情况调整
    ax.grid(True, linestyle="--", alpha=0.5)

    # 初始化四条曲线（返回线对象，用于后续更新）
    train_loss_line, = ax.plot([], [], color='red', label="train_loss", linewidth=2)
    test_loss_line, = ax.plot([], [], color='orange', label="test_loss", linewidth=2)
    train_acc_line, = ax.plot([], [], color='green', label="train_acc", linewidth=2)
    test_acc_line, = ax.plot([], [], color='blue', label="test_acc", linewidth=2)

    ax.legend()
    plt.tight_layout()

    return fig, ax, [train_loss_line, test_loss_line, train_acc_line, test_acc_line]


def update_dynamic_plot(ax, lines, epoch, train_loss, test_loss, train_acc, test_acc):
    """更新图表数据"""
    # 提取线对象
    train_loss_line, test_loss_line, train_acc_line, test_acc_line = lines

    # 更新每条线的数据（x为已迭代的epoch，y为对应的指标）
    x = list(range(1, epoch + 1))
    train_loss_line.set_data(x, train_loss[:epoch])
    test_loss_line.set_data(x, test_loss[:epoch])
    train_acc_line.set_data(x, train_acc[:epoch])
    test_acc_line.set_data(x, test_acc[:epoch])

    # 刷新图表
    ax.draw_artist(ax.patch)  # 高效重绘背景
    for line in lines:
        ax.draw_artist(line)
    ax.figure.canvas.flush_events()  # 刷新画布


class DynamicTrainingPlot:
    # 动态训练指标可视化类，用于实时展示训练/测试的准确率和损失曲线"""
    def __init__(self, total_epochs, title="training_target"):
        # 初始化初始化动态图表，用于存储历史数据
        self.total_epochs = total_epochs  # 总训练轮次
        self.train_loss_history = []      # 训练损失历史
        self.test_loss_history = []       # 测试损失历史
        self.train_acc_history = []       # 训练准确率历史
        self.test_acc_history = []        # 测试准确率历史

        # 设置中文字体
        plt.rcParams["font.family"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False

        # 启用交互模式
        plt.ion()

        # 创建图表和轴对象
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.set_title(title)
        self.ax.set_xlabel("epoch")
        self.ax.set_ylabel("acc/loss")
        self.ax.set_xlim(1, total_epochs)  # x轴范围固定为总迭代次数
        self.ax.set_ylim(0, 1.2)          # y轴范围（适合acc和loss，loss可能超过1时需调整）
        self.ax.grid(True, linestyle="--", alpha=0.5)

        # 初始化四条曲线（线对象用于后续更新）
        self.train_loss_line, = self.ax.plot([], [], color='red', label="训练损失", linewidth=2)
        self.test_loss_line, = self.ax.plot([], [], color='orange', label="测试损失", linewidth=2)
        self.train_acc_line, = self.ax.plot([], [], color='green', label="训练准确率", linewidth=2,linestyle="--")
        self.test_acc_line, = self.ax.plot([], [], color='blue', label="测试准确率", linewidth=2,linestyle="--")

        # 添加图例和布局调整
        self.ax.legend()
        plt.tight_layout()

    def update(self, epoch, train_loss, test_loss, train_acc, test_acc):
        """
        更新图表数据

        参数:
        epoch: 当前轮次（从1开始）
        train_loss: 当前训练损失
        test_loss: 当前测试损失
        train_acc: 当前训练准确率
        test_acc: 当前测试准确率

        """
        # 存储当前轮次的指标
        self.train_loss_history.append(train_loss)
        self.test_loss_history.append(test_loss)
        self.train_acc_history.append(train_acc)
        self.test_acc_history.append(test_acc)

        # 生成x轴数据（已训练的轮次）
        x = list(range(1, epoch + 1))

        # 更新每条曲线的数据
        self.train_loss_line.set_data(x, self.train_loss_history)
        self.test_loss_line.set_data(x, self.test_loss_history)
        self.train_acc_line.set_data(x, self.train_acc_history)
        self.test_acc_line.set_data(x, self.test_acc_history)

        # 高效刷新图表
        self.ax.draw_artist(self.ax.patch)  # 重绘背景
        for line in [self.train_loss_line, self.test_loss_line,
                    self.train_acc_line, self.test_acc_line]:
            self.ax.draw_artist(line)
        self.fig.canvas.flush_events()  # 刷新画布

    def close(self):
        """
        关闭交互模式，保存图表（可选）
        """
        plt.ioff()  # 关闭交互模式
        # plt.savefig("training_curve.png")  # 可选：保存最终图表
        plt.show()  # 显示最终图表（阻塞程序，直到关闭窗口）


# 模拟训练过程
if __name__ == "__main__":
    # 假设训练总轮次为20
    total_epochs = 20

    # 创建动态图表实例
    dynamic_plot = DynamicTrainingPlot(
        total_epochs=total_epochs,
        title="MNIST模型训练曲线"  # 可自定义标题
    )

    # 模拟训练循环（实际使用时替换为真实训练逻辑）
    for epoch in range(1, total_epochs + 1):
        # 模拟当前轮次的指标（实际训练中替换为真实计算的loss和acc）
        train_loss = 0.5 * (1 - epoch / total_epochs)  # 模拟损失下降
        test_loss = 0.6 * (1 - epoch / total_epochs)
        train_acc = 0.8 + 0.18 * (epoch / total_epochs)  # 模拟准确率上升
        test_acc = 0.75 + 0.15 * (epoch / total_epochs)

        # 更新动态图表
        dynamic_plot.update(
            epoch=epoch,
            train_loss=train_loss,
            test_loss=test_loss,
            train_acc=train_acc,
            test_acc=test_acc
        )

    # 训练结束后关闭图表
    dynamic_plot.close()

