# 引言

本文分成三个部分
+ 之前的研究——透明物体位姿估计
+ 当前正在研究——普通物体的位姿估计（简历上所写的部分）
+ 数据集的采集和优化

# 刚体三维位姿估计

刚体三维位姿估计是从RGB或RGBD数据中预测出物体在相机坐标系下的三自由度旋转和三自由度位移，这对于机器人抓取和AR领域具有重要支撑作用。

<img src=".\assert\15.jpg" width="1500px" align="center" style="zoom:50%;" />

目前物体三维位姿估计存在着许多的挑战，大部分方法依赖深度数据，但是当面对透明和高反光物体的时候，光学特性的3D传感器难以对齐进行准确的深度估计，因此本研究的重点在于如何解决对于透明和高反光物体以及正常物体的泛化性方法研究。

## 1. 透明物体位姿估计
### 1.1 透明物体6D Pose预测研究成果
> Xu C, Chen J, **Yao M,** et al. 6dof pose estimation of transparent object from a single rgb-d image[J]. Sensors, 2020, 20(23): 6790.

面对透明物体存在的反光、低纹理、透明等视觉特性，深度传感器难以对齐进行有效估计。我们这篇文章的思路是利用其表面法向量和UV坐标来补齐深度特征，从而实现对透明物体较好的预测。

但是存在以下问题：

+ 需要桌面进行支撑，在没有背景的情况下难以预测
+ 效果有待提升

### 1.2 后续工作
#### 1.2.1 改进——多模态隐式特征融合
针对上面的问题我提出了一个新的模型

<img src=".\assert\02.png" align="center" style="zoom:150%;" />

去掉对于背景深度信息的依赖，依靠近大远小的原则让网络去学习到**绝对深度信息**，**相对深度信息**利用表面法向量特征进行补全，得到隐式的三维特征，然后融合RGB特征和三维特征利用全卷积层得到最终的6D位姿。

对于物体对称带来的旋转多解的情况，我们利用旋转矩阵的特性来对其进行约束。



<img src=".\assert\04.png" width="960px" style="zoom:50%;" />

#### 1.2.2 结果分析
在[Cleargrasp](https://sites.google.com/view/cleargrasp)数据集上的定量分析，[详细结果](https://github.com/yaomy533/pose_estimation/blob/master/version/transparent/eval_log.txt)，其中cup、square、stemless都是具有轴对称的物体。

<img src=".\assert\05.png" width="1800px" align="center" style="zoom:30%;" />

部分结果展示：

<img src=".\assert\03.png" width="960px" align="center" style="zoom:50%;" />





## 2. 普通物体的位姿估计（当前正在研究）

### 2.1 代码结构介绍
本仓库所带的代码就是这部分内容，代码结构介绍：
+ dataset: 数据集加载和预处理代码
+ lib

  + network：模型(krrn.py)以及损失函数(loss.py)
  
  + transform: 坐标变换工具代码
  
  + utils：log，metic等工具代码
  
+ tool

  + script: 一些处理脚本
  
  + viz: jupyter 可视化代码（用于远程可视化调试）
  
  + trainer.py: 训练和测试过程封装的类
  
+ train.py: 主函数 


### 2.2 方法介绍
<img src=".\assert\10.png" width="960px" align="center" style="zoom:50%;" />

+ 鲁棒特征提取：利用多分辨率特征提取网络HRNet来提取彩色特征，逐像素重建物体模型、预测表面法向量、掩码；
+ 密集多模态特征融合：将重建出物体模型、表面法向量等从RGB图像中得到的特征，同深度相机得到的三维点云的特征，**利用PointNet++的思想进行多次层、多尺度融合**，得到三维语义信息丰富的特征；
+ 解耦旋转和位移的预测：采用PnPRANSAC来从重建的模型获取三维旋转R，以及利用融合特征来预测T在深度图重建的三维点云的偏移量得到T，以此来**解耦6D Pose的预测**，并将预测结果利用旋转矩阵特性以及K-近邻算法解决物体对称性的问题；
+ 在公有数据集取得不错的成果；

> Wang C, Xu D, Zhu Y, et al. Densefusion: 6d object pose estimation by iterative dense fusion[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019: 3343-3352.

同Densefusion对比结果

<img src=".\assert\12.png" width="1500px" align="center" style="zoom:50%;" />

## 3. 数据集的采集和优化

### 3.1 真实数据集的采集

采集目标：低纹理、高反光、透明物体混合场景，获取物体的三维模型，并半自动精确标注物体的6D位姿。

+ CAD模型采集

<img src=".\assert\06.png" width="480px" align="center" style="zoom:50%;" />

+ 半自动数据采集和标注流程

<img src=".\assert\11.png" width="640px" align="center" style="zoom:50%;" />

> 因为我们实验室的机械臂好久没有矫正，所以读出来的位姿出现了误差，或者大部分机械臂都有这个误差，因为目前而言还没有用机械臂采集出来的6D Pose数据集，可能用机械臂进行采集本身就有这些误差。
> 从我们实验的结果可以保证机械臂的相对精度，也就是往返能够保证准确，但是如果是读数来算精度就不足以支撑我们的数据采集。

根据上面的摸索，我们最终确认的数据采集流程：

+ 利用机械臂采集多视角标定板图片（约1000张），根据标定板我们可以得到每张图片对应的相机位姿。
+ 我们控制机械臂原路返回，再拍一组带实物的图片（约1000张），由于两个走的是相同的路径，我们采用KNN来计算两次机械臂记录的相近的位姿，根据利用标定板得到的位姿**获取每张图片之间的相对位姿**。

+ 再从中选出四个视角，**手动调整**多视角对齐来**获取绝对位姿**，再根据上述计算出的相对位姿，**自动计算**最终1000张图片的位姿。
+ 处理一组1000张图片，整个标注过程不超过10min，包含计算时间。

<img src=".\assert\07.png" width="960px" align="center" style="zoom:50%;" />

自动计算标注结果展示：

<img src="https://github.com/yaomy533/pose_estimation/blob/master/assert/8.gif" />

<img src="https://github.com/yaomy533/pose_estimation/blob/master/assert/9.gif" />

### 3.2 对已有数据集的标注

面临场景：已有一系列已经采集完成的人手数据视频流，对于一帧有四个视角，但是没有标定其6D位姿。

首先我们利用在公有数据集上训练你的MaskRCNN来预测视角$i$下的人手掩码 $\hat m_i$ 和利用OpenPose预测其二维人手关键点 $\hat k_i$，相机内参$K$已知，人手模型为$P$，我们的任务可以描述为：

<img src=".\assert\14.png" width="1500px" align="center" style="zoom:50%;" />

理论上这是一个凸优化过程，因为利用公有数据集训练出的MaskRCNN和OpenPose模型对于我们人手的数据集会预测不准确，同时因为相机标注、多相机同步等问题也会带来误差，所以是非凸的。

因此我们采用的方法是：

+ 对可能的位姿空间进行蒙特卡洛均匀采样，利用遗传算法来获得精英位姿，得到Coarse位姿。
+ 对Coarse位姿进行梯度下降，得到精确的位姿Refine。

+ 对于相邻帧，因为时间序列的关系，其位姿变化不大，可直接采用梯度下降来优化得到结果。

最终得到的位姿，将其重投影到多视角平面：

<img src=".\assert\13.png" width="1500px" align="center" style="zoom:50%;" />





