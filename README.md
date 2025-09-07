# Pytorch-MNIST-NUMBER-Fashion-MNIST-LeNet-MLP
这是我的Pytorch深度学习的代码库，有一些经典的数据集和模型，可供初学者参考

+ 点击number_minist.py文件里的这里可以运行代码
<img width="678" height="181" alt="image" src="https://github.com/user-attachments/assets/61aeff4e-c275-42c7-8a93-dbfc04ac9e03" />

## 开发环境
+ conda环境
+ python版本最好在3.8到3.10直接
+ 根据import安装对应的库即可，常用的有numpy、matplotlib、tqdm、pytorch、opencv等
+ pytorch官网下载带cuda的pytorch：https://pytorch.org/

## 每个py文件的介绍：
+ data文件：数据集存储地
+ flask_show.py:尝试将模型通过flask部署到网页前端，但由于技术力不够，已放弃，即该文件无用
+ models.py：模型存储文件，均继承自父类nn.Moudle，实现了多层感知机和LeNet还有AI魔改的卷积神经网络
+ number_minist.py：模型运行文件，只需在该文件运行得到模型训练过程，其它功能看注释
+ opencv_process.py：用来测试opencv处理图像的文件，自定义处理函数process，但已在number_minist中定义，对模型训练无用
+ plt_show.py：用matplotlib打印数据集的前几张图像，但使用的部分已被删除，需要展示的自行阅读添加
+ test.py：测试语法文件，无用

## 运行效果图
<img width="1880" height="971" alt="92b6701dbff4361fca8deeb2e71b6693" src="https://github.com/user-attachments/assets/ce1e65f6-6a1a-43df-8c4c-59d4dddbec93" />



