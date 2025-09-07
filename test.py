import torch

# 模拟经过 ToTensor() 和 x*255 后的图像张量（单通道，1×28×28）
img = torch.randint(0, 256, (1, 28, 28), dtype=torch.float32)
print("处理前形状:", img)  # 输出：torch.Size([1, 28, 28])

# 应用你的 process 函数
def process(img):
    img[img > 128] = 255
    img[img <= 128] = 0
    return img

processed_img = process(img)
print("处理后形状:", processed_img)  # 输出：torch.Size([1, 28, 28])（形状不变）