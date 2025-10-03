# Pytorch-MLP-LeNet-Are-Used-To-Train-Mnist-Model

## 一、项目介绍
+ 本项目基于Pytorch构建了多层感知机(MLP)与一个简单的神经网络(LeNet)，采用mnist（手写数字识别）数据集进行训练与预测，该数据集是众多人工智能学习者第一个接触到的数据集，对于理解深度学习的代码有重要作用。
+ 本项目适合深度学习的初学者，尤其适合在学习了多层感知机和卷积神经网路基础(大多是LeNet入门)后，知道原理但不知道如何写代码的初学者。
+ 本项目只包含简单的模型构建与训练、测试过程，没有复杂的部署与可视化模块。
+ 流程图(重复部分请不要在意，为制图失误)：

![手写数字识别效果示意图](https://github.com/zlyd-CV/Photos_Are_Used_To_Others_Repository/blob/abf872779e8d74c0f8871642634f2f46205678db/Simple-Pytorch-MLP-LeNet-Are-Used-To-Train-Mnist-Model/%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E6%B5%81%E7%A8%8B%E5%9B%BE.svg)

## 二、内容介绍
+ 本项目包含：
+ requirements.txt：包的版本，运行下面命令即可下载到虚拟环境中，pytorch请前往官网下载
 ```txt
pip install -r requirements.txt
```
  + data文件夹：数据集存放地址
  + log文件夹：tensorboard写入事件的地址
  + models文件夹：保存模型(参数+计算图)的地址
  + main.py：主程序，运行该文件即可运行整个项目
  + models.py：构建模型类
  + plt_show：matplotlib可视化训练指标变化
  + test_version：检查包的版本号（简单项目中用处不大）

## 三、运行展示
![运行效果图](https://github.com/zlyd-CV/Photos_Are_Used_To_Others_Repository/blob/abf872779e8d74c0f8871642634f2f46205678db/Simple-Pytorch-MLP-LeNet-Are-Used-To-Train-Mnist-Model/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-09-27%20132525.png)
![运行效果图](https://github.com/zlyd-CV/Photos_Are_Used_To_Others_Repository/blob/abf872779e8d74c0f8871642634f2f46205678db/Simple-Pytorch-MLP-LeNet-Are-Used-To-Train-Mnist-Model/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-09-27%20132805.png)

## 四、部分资源下载地址
+ pytorch官网下载带cuda的pytorch：https://pytorch.org/


