from torchvision import datasets, transforms
from plt_show import *
from models import *
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

data_path = r"D:\pycharmprofession\Programme\Pycharm_Project\pytorch learning\pytorch learning\data"
# 为后续将模型训练移植到GPU上做准备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter("log") # 写入tensorboard

if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])

    # 下载训练集
    train_data = datasets.MNIST(root=data_path, train=True, transform=transform, download=True)
    # 下载测试集（若需训练，同理加载训练集）
    test_data = datasets.MNIST(root=data_path, train=False, transform=transform, download=True)
    print(f"训练集的长度是:{len(train_data)}")
    print(f"测试集的长度是:{len(test_data)}")

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False)

    # 可调节的超参数
    batch_size = 128
    epochs = 20
    # 实例化模型
    model = LeNet()
    model = model.to(device)
    # 定义动态绘制图标类
    dynamic_plot = DynamicTrainingPlot(
        total_epochs=epochs,
    )
    # 损失函数：交叉熵损失
    loss_func = nn.CrossEntropyLoss()
    # 优化函数：Adam优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # 向tensorboard写入模型结构（打开命令:tensorboard --logdir=log）
    init_img = torch.zeros((128, 1, 28, 28), device=device)  # 将输入搬到模型训练的地方
    writer.add_graph(model, init_img)  # 得到计算图

    # 训练集训练
    for epoch in tqdm(range(1, epochs + 1), desc="【整体训练进度】", ncols=100, unit="epoch", position=0, leave=True):
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

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            # 统计训练精度
            train_total_loss += loss_train.item()
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

        tqdm.write(f"\n第{epoch}次epoch的训练集acc为：{train_acc}，LOSS为：{avg_train_loss}")
        tqdm.write(f"第{epoch}次epoch的测试集acc为：{test_acc}，LOSS为：{avg_test_loss}")
        # 更新动态图表
        dynamic_plot.update(
            epoch=epoch,
            train_loss=avg_train_loss,
            test_loss=avg_test_loss,
            train_acc=train_acc,
            test_acc=test_acc
        )
    # 训练结束后关闭图表
    dynamic_plot.close()
    # torch.save(model, "models/ResAttCNN_FashionMNIST-50.pkl")
