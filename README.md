# 本文分成三个部分
+ 之前研究—透明物体位姿估计
+ 当前正在研究—普通物体的位姿估计（简历上所写的部分）
+ 数据集的采集和优化

# 刚体三维位姿估计

目前物体三维位姿估计存在着许多的挑战，大部分方法依赖深度数据，但是当面对透明和高反光物体的时候，光学特性的3D传感器难以对齐进行准确的深度估计，因此本研究的重点在于如何解决对于透明和高反光物体以及正常物体的泛华性方法研究。

### 1. 之前研究—透明物体位姿估计

> Xu C, Chen J, **Yao M,** et al. 6dof pose estimation of transparent object from a single rgb-d image[J]. Sensors, 2020, 20(23): 6790.

面对透明物体存在的反光、低纹理、透明等视觉特性，深度传感器难以对齐进行有效估计。我们这片文章的思路是利用其表面法向量和UV坐标来补齐深度特征，从而实现对透明物体较好的预测。

但是存在以下问题：

+ 需要桌面进行支撑，在没有背景的情况下难以预测
+ 效果有待提升

<img src=".\assert\01.png" align="center" width="640" style="zoom: 75%;" />

针对上面的问题我提出了一个新的模型

<img src=".\assert\02.png" align="center" style="zoom:150%;" />

与上面的不同，我们不再需要深度信息，而是只从单张RGB特征图中利用人为制作的表面法向量和深度特征（训练时）的约束来让网络具有丰富的几何特征学习能力。并且对于不同源的特征，参考PointNet我们采用了一种逐像素融合的融合方法，通过平均池化从特征中提取全局信息，并全局特征加入到每一个像素中去，这样我们就拥有语义丰富的三维几何特征，后面利用一个MLP网络从容和特征中回归出位姿。

此外对于物体对称带来的旋转多解的情况，我们利用旋转矩阵的特性来对其进行约束。

<img src=".\assert\04.png" width="480px" style="zoom:50%;" />



> Sajjan S, Moore M, Pan M, et al. Clear grasp: 3d shape estimation of transparent objects for manipulation[C]//2020 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2020: 3634-3642.

在Cleargrasp数据集上的定量分析，详细结果[点这里](https://github.com/yaomy533/pose_estimation/blob/master/version/transparent/eval_log.txt)，其中heart、square、stemless都是具有轴对称的物体。
<img src=".\assert\05.png" width="640px" align="center" style="zoom:30%;" />

估计位姿的定性分析：

<img src=".\assert\03.png" width="960px" align="center" style="zoom:50%;" />

## 2. 当前正在研究—普通物体的位姿估计

<img src=".\assert\10.png" width="640px" align="center" style="zoom:50%;" />

+ 鲁棒特征提取：利用多分辨率特征提取网络HRNet来提取彩色特征，逐像素重建物体模型、预测表面法向量、掩码；
+ 密集多模态特征融合：将重建出物体模型、表面法向量等从RGB图像中得到的特征，同深度相机得到的三维点云的特征，**利用PointNet++的思想进行多次层、多尺度融合**，得到三维语义信息丰富的特征；
+ 解耦旋转和位移的预测：采用PnPRANSAC来从重建的模型获取三维旋转R，以及利用融合特征来预测T在深度图重建的三维点云的偏移量得到T，以此来**解耦6D Pose的预测**，并将预测结果利用旋转矩阵特性以及K-近邻算法解决物体对称性的问题；
+ 在公有数据集取得不错的成果；

> Wang C, Xu D, Zhu Y, et al. Densefusion: 6d object pose estimation by iterative dense fusion[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019: 3343-3352.

同Densefusion对比结果

| 物体 | DenseFusion | OURS（RGB-D, 不带Refine） |
| :--: | :---------: | :-----------------------: |
|  1   |    92.3     |            85             |
|  2   |    93.2     |         **98.6**          |
|  4   |    94.4     |         **98.0**          |
|  5   |    93.1     |         **99.1**          |
|  6   |    96.5     |         **98.6**          |
|  8   |    87.0     |         **96.4**          |
|  9   |    92.3     |            90             |
|  10  |    99.8     |          **100**          |
|  11  |     100     |           99.8            |
|  12  |    92.1     |         **94.3**          |
|  13  |    97.0     |           96.3            |
|  14  |    95.3     |         **98.2**          |
|  15  |    92.8     |         **98.1**          |
| 平均 |    94.3     |           97.1            |

## 3. 数据集的采集和优化

### 3.1 真实数据集的采集

采用三维扫描仪来获取物体的CAD模型，我们采用的是工业级的高精度三维扫描仪，扫描精度达到0.2mm，足够支撑我们对于位姿预测的研究，下图是我们用扫描仪得到的物体模型（部分）。

<img src=".\assert\06.png" width="640px" align="center" style="zoom:50%;" />

利用不精确机械臂，来进行半自动化标注，标注流程如下：

> 因为我们实验室的机械臂好久没有矫正，所以读出来的位姿出现了误差，或者大部分机械臂都有这个误差，因为目前而言还没有用机械臂采集出来的6D Pose数据集，可能用机械臂进行采集本身就有这些误差。
> 从我们实验的结果可以保证机械臂的相对精度，也就是往返能够保证准确，但是如果是读数来算精度就不足以支撑我们的数据采集。

+ 利用机械臂采集多视角标定板图片（约1000张），根据标定板我们可以得到每张图片的位姿
+ 我们再往返拍一组带实物的图片（约1000张），根据这两组图片计算出其相对位姿。

+ 再从中选出四个视角，**手动调整**多视角对齐来获取绝对位姿，再根据上述计算出的相对位姿，**自动计算**最终1000张图片的位姿。
+ 上述一组过程标注大概耗时10min左右。

手动调整过程展示：

<img src=".\assert\07.png" width="640px" align="center" style="zoom:50%;" />

自动计算标注结果展示：

<img src="https://github.com/yaomy533/pose_estimation/blob/master/assert/8.gif" />

<img src="https://github.com/yaomy533/pose_estimation/blob/master/assert/9.gif" />

### 3.2 对已有数据集的标注

这个工作面临的场景是我们已经采集出了一组人手的数据集，但是没有标注其姿态，我们希望利用自动标注的方法来获得其6D Pose。

数据集介绍：多视角人手数据集，总共有四个视角，四个视角的位置已知，其他未知。

目的：获取人手的6D Pose。

+ 首先利用OpenPose和MaskRCNN（公有数据集训练出的模型）来对我们的数据预测Mask和关节点（二维关节点），因为这两个都不是在我们数据集上运算的到的结果，所以这两种数据预测的结果存在着一定的误差。我们这个工作的难点如何从这种带误差的标注中获取我们想要的位姿。
+ 我们采用遗传算法（蒙特卡洛采样）的Coarse优化和梯度下降的精确优化；具体过程可以参考[文档]([version/人手数据集优化/实验.md](https://github.com/yaomy533/pose_estimation/blob/master/version/%E4%BA%BA%E6%89%8B%E6%95%B0%E6%8D%AE%E9%9B%86%E4%BC%98%E5%8C%96/%E5%AE%9E%E9%AA%8C.md))
+ 从图中的[四个视角](https://github.com/yaomy533/pose_estimation/blob/master/version/%E4%BA%BA%E6%89%8B%E6%95%B0%E6%8D%AE%E9%9B%86%E4%BC%98%E5%8C%96/%E5%AE%9E%E9%AA%8C%E5%9B%BE/%E6%96%B9%E6%A1%88%E5%9B%9B/GIF20210925161019.gif)可以看到我们对于位姿（旋转和平移）已经比较准确了，之所以会存在误差是因为**人手的建模存在误差**，所以导致看着好像不准确。



