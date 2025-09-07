import cv2 as cv
import numpy as np
import torch
from torch.onnx.symbolic_opset9 import permute
from torchvision import datasets, transforms

"""
opencv处理图像
实验目的：为了验证一些简单的图像变换是否能增强模型的线性效果                       
主要操作：去噪(滤波)、灰度变换、直方图修正，即实现图像对比度增强或者图像平滑
"""

data_path = r"D:\zlyd\pycharm\project\pytorch learning\data"


def process(img):
    img[img > 128] = 255
    img[img <= 128] = 0
    return img


if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor()])
    # 下载训练集
    train_data = datasets.MNIST(root=data_path, train=True, transform=transform, download=True)
    # 下载测试集（若需训练，同理加载训练集）
    test_data = datasets.MNIST(root=data_path, train=False, transform=transform, download=True)

    batch_size = 1

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    # 迭代数据并显示
    for images, labels in train_loader:
        # 1. 取第一个样本（移除批次维度）
        img_tensor = images[0]  # 形状: [1, 28, 28]（单通道图像）

        # 2. 转换为NumPy数组，并调整维度顺序
        # 对于单通道图像：[C, H, W] -> [H, W]
        img_np = img_tensor.squeeze().numpy()  # squeeze() 移除通道维度

        # 3. 可选：MNIST图像是0-1范围，转换为0-255便于显示
        img_np = (img_np * 255).astype('uint8')
        print(img_np, type(img_np))
        img_np = cv.resize(img_np, (256, 256))

        # 4. 显示图像
        cv.imshow("MNIST Image", img_np)
        print(img_np.shape)
        print(f"标签: {labels[0].item()}")  # 打印对应标签
        img_process=process(img_np)
        cv.imshow("img_process", img_process)
        cv.waitKey(0)


        break  # 只显示第一个样本

    cv.destroyAllWindows()
