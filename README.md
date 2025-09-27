# Introduction-Pytorch-MLP-LeNet-Are-Used-To-Train-Mnist-Model
## 一、事先声明
+ 在您进入到本项目前，作者先对本项目难度做一个解释，作者本人所有项目难度划分运用以下规则：
  + Introduction:入门级别，不包含模型部署只负责训练，只提供少量的可视化功能，模型大多为简单模型(介于ResNet与U-Net难度)。
  + Intermediate:中等级别，大部分项目不包含或少部分项目只包含简单的前端部署，模型大多为2016年之后提出的。
  + Advanced:困难级别，基本上包含模型部署模块，模型多为2020年后提出的新时代模型架构(例如Mamba)。
  + Competition/Thesis:个人参与学术竞赛与发表论文的项目，出于部分原因可能项目会缺失数据集等。

## 二、项目介绍
+ 本项目基于Pytorch构建了多层感知机(MLP)与一个简单的神经网络(LeNet)，采用mnist（手写数字识别）数据集进行训练与预测，该数据集是众多人工智能学习者第一个接触到的数据集，对于理解深度学习的代码有重要作用。
+ 本项目适合深度学习的初学者，尤其适合在学习了多层感知机和卷积神经网路基础(大多是LeNet入门)后，知道原理但不知道如何写代码的初学者。
+ 本项目只包含简单的模型构建与训练、测试过程，没有复杂的部署与可视化模块。
+ 流程图：

![手写数字识别效果示意图](https://github.com/zlyd-CV/Photos_Are_Used_To_Others_Repository/blob/ba2f7f23e1f71c505679808ed7bbe21890253f17/Introduction-Pytorch-MLP-LeNet-Are-Used-To-Train-Mnist-Model/%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E6%B5%81%E7%A8%8B%E5%9B%BE.svg)

## 三、内容介绍
+ 本项目包含：
  + data文件夹：数据集存放地址
  + log文件夹：tensorboard写入事件的地址
  + models文件夹：保存模型(参数+计算图)的地址
  + main.py：主程序，运行该文件即可运行整个项目
  + models.py：构建模型类
  + plt_show：matplotlib可视化训练指标变化
  + test_version：检查包的版本号（简单项目中用处不大）

## 四、运行展示
<img width="2418" height="1198" alt="image" src="https://github.com/user-attachments/assets/643957d7-e9f0-48ce-a53c-5f49fcb53ad7" />
<img width="1394" height="1123" alt="image" src="https://github.com/user-attachments/assets/3402d2f1-edc5-41a3-85fa-5e2348029a4d" />



## 五、部分资源下载地址
+ pytorch官网下载带cuda的pytorch：https://pytorch.org/


