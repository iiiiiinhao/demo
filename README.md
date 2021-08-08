# demo
An FCN network model for 3D image registration

***文件说明***
### model_aspp 为基于无监督学习的肺3D-CT非刚性配准模型文件
其中包括conv_down、 PPM、 ASPP、 SElayer 共4个模组class，以及1个模型网络类 Net， 一个snet用于加载模型。
将一对三维图像对（浮动图像，固定图像）输入model，通过下采样、提取特征，预测像素移动方向、距离后，得到一个密集形变矢量场（DVF），
DVF描述了浮动图像相对于固定图像各像素点的形变矢量。

### warp 为3D图像形变文件 ###
得到DVF后，需要将浮动图像输入warp模块，最终得到配准图像。
首先创建一个等同于图像大小的网格，将DVF数据（flow）加载进入网格，再通过线性插值得到最终配准图像。
