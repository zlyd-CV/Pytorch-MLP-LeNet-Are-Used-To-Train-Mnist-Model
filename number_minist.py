import copy

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from plt_show import *
from models import *
from tqdm import tqdm
import cv2 as cv

data_path = r"D:\zlyd\pycharm\project\pytorch learning\data"
# 为后续将模型训练移植到GPU上做准备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process(img):
    """
    对图像进行预处理：转为灰度图并应用直方图均衡化，返回PyTorch Tensor。
    支持输入：PyTorch Tensor（[0,1]）或 PIL图像（[0,255]）。
    返回：PyTorch Tensor（[0,1]，形状为(1, H, W)）
    """
    # 处理Tensor输入（来自ToTensor()，已归一化到[0,1]）
    if isinstance(img, torch.Tensor):
        # 去除通道维度（假设输入形状为(1, H, W)），转为[0,255]的numpy数组
        img_np = (img.squeeze(0).numpy() * 255).astype(np.uint8)
    # 处理PIL图像输入（未归一化，范围[0,255]）
    else:
        img_np = np.array(img, dtype=np.uint8)
        # 若PIL图像是单通道，转为(1, H, W)会被转换为(H, W)，无需额外处理
    # 确保是单通道灰度图（若输入是多通道）
    if len(img_np.shape) == 3:
        # 假设输入是RGB格式（PIL图像默认），转为灰度图
        img_np = cv.cvtColor(img_np, cv.COLOR_RGB2GRAY)
    # 应用直方图均衡化（仅支持单通道）
    img_processed=img_np*1.33
    img_processed.clip(0,255)
    # 转回Tensor：添加通道维度(1, H, W)，并归一化到[0,1]
    return torch.from_numpy(img_processed).unsqueeze(0).float() / 255.0


# 初始化线性层的权重与偏置
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
        nn.init.zeros_(m.bias)


if __name__ == "__main__":
    """
          transforms.Lambda(lambda x: x * 255),  # 还原到[0,255]范围
          transforms.Lambda(process),  # 应用自定义处理函数
          transforms.Lambda(lambda x: x / 255)  # 再次归一化到[0,1]范围，适合模型输入
    """

    transform = transforms.Compose([
        transforms.ToTensor(),  # 转为Tensor并归一化到0-1
        transforms.Lambda(lambda x: process(x)),  # 在process函数内部处理所有转换
    ])

    transform1 = transforms.Compose([transforms.ToTensor()])

    # 下载训练集
    train_data2 = datasets.FashionMNIST(root=data_path, train=True, transform=transform1, download=True)
    train2_loader = torch.utils.data.DataLoader(train_data2, batch_size=64, shuffle=True)

    # 下载训练集
    train_data = datasets.FashionMNIST(root=data_path, train=True, transform=transform, download=True)
    # 下载测试集（若需训练，同理加载训练集）
    test_data = datasets.FashionMNIST(root=data_path, train=False, transform=transform, download=True)
    print(f"训练集的长度是:{len(train_data)}")
    print(f"测试集的长度是:{len(test_data)}")

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=True)

    for index, (img, label) in tqdm(enumerate(train2_loader)):
        img_temp1 = img[index].squeeze().numpy()
        img_np = (img_temp1 * 255).astype('uint8')
        print(index, img_temp1.shape)
        img_temp2 = process(img_np)
        img_temp2 = img_temp2.squeeze().numpy()
        # 先创建副本再调整大小
        img_temp11 = cv.resize(img_temp1, (256, 256), interpolation=cv.INTER_AREA)
        img_temp22 = cv.resize(img_temp2, (256, 256), interpolation=cv.INTER_AREA)
        cv.imshow("img_temp1", img_temp11)
        cv.imshow("img_temp2", img_temp22)
        cv.waitKey(0)
        break

    """
    first_batch = next(iter(train_loader))
    batch_images, batch_labels = first_batch
    show_mnist_images(batch_images, batch_labels, num_show=10)
    """

    # 可调节的超参数
    batch_size = 256
    epochs = 3
    # 定义CNN类
    model = ResAttCNN()
    model = model.to(device)
    # 定义动态绘制图标类
    dynamic_plot = DynamicTrainingPlot(
        total_epochs=epochs,
        title="fashion——mnist数据集"
    )
    # 损失函数：交叉熵损失
    loss_func = nn.CrossEntropyLoss()
    # 优化函数：Adam优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 训练集训练
    for epoch in tqdm(range(epochs), desc="【整体训练进度】", ncols=100, unit="epoch", position=0, leave=True):
        loss_train, loss_test = 0.0, 0.0
        train_total_loss, test_total_loss = 0.0, 0.0
        train_total_correct = 0  # 训练集总正确样本数
        train_total_samples = 0  # 训练集总样本数
        model.train()
        for index, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            # 前向传播操作
            result = model(images)
            # 传入输出层节点和真实标签来计算损失函数
            loss_train = loss_func(result, labels)
            # 先清空梯度再反向传播
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            train_total_loss += loss_train.item()
            # 统计训练精度
            pred_labels = torch.argmax(result, dim=1)  # 取概率最大的类别作为预测结果
            batch_correct = (pred_labels == labels).sum().item()  # 当前批次正确样本数
            train_total_correct += batch_correct
            train_total_samples += images.size(0)

        avg_train_loss = train_total_loss / len(train_loader)  # 平均训练损失
        train_acc = train_total_correct / train_total_samples  # 训练集正确率

        # 测试集验证a
        model.eval()
        test_total_correct = 0  # 测试集总正确样本数
        test_total_samples = 0  # 测试集总样本数
        with torch.no_grad():
            for index, (images, labels) in enumerate(test_loader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss_test = loss_func(outputs, labels)
                test_total_loss += loss_test.item()
                # 统计测试精度
                pred_labels = torch.argmax(outputs, dim=1)
                batch_correct = (pred_labels == labels).sum().item()
                test_total_correct += batch_correct
                test_total_samples += images.size(0)

            avg_test_loss = test_total_loss / len(test_loader)
            test_acc = test_total_correct / test_total_samples

        tqdm.write(f"\n第{epoch + 1}次epoch的训练集acc为：{train_acc}，LOSS为：{avg_train_loss}")
        tqdm.write(f"第{epoch + 1}次epoch的测试集acc为：{test_acc}，LOSS为：{avg_test_loss}")
        # 更新动态图表
        dynamic_plot.update(
            epoch=epoch + 1,
            train_loss=avg_train_loss,
            test_loss=avg_test_loss,
            train_acc=train_acc,
            test_acc=test_acc
        )
    # 训练结束后关闭图表
    dynamic_plot.close()
    torch.save(model, "models/ResAttCNN_FashionMNIST-100.pkl")
