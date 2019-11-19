# Pix2Pose:Pixel-Wise Coordinate Regression of Objects for 6D Pose Estimation

> <font
> size=4> 论文地址：http://openaccess.thecvf.com/content_ICCV_2019/papers/Park_Pix2Pose_Pixel-Wise_Coordinate_Regression_of_Objects_for_6D_Pose_Estimation_ICCV_2019_paper.pdf
> <font
> size=4> github链接：https://github.com/kirumang/Pix2Pose

## 简介

<font size=4> &#160; &#160; &#160; &#160;作者主要解决的是遮挡、对称和无纹理问题，提出一种新的位姿估计方法：pix2pose，可以在没有纹理模型的情况下预测每个目标像素的三维坐标，设计了一种自动编码器结构来估计三维坐标和每个像素的期望误差。使用像素级的预测生成2D-3D之间的对应，从而直接使用RANSAC迭代的PnP算法计算位姿。这种方法利用生成对抗训练来精确的重新覆盖被遮挡部分，从而达到对遮挡的鲁棒性。同时提出了一种新的损失函数：transformer loss，引导预测位姿到最接近的对称位姿来处理对称物体。

## 之前的工作介绍

<font size=4> &#160; &#160; &#160; &#160;虽然深度图能够提供精确的三维像素坐标，但是获得深度图并不容易。大量的位姿估计依赖于物体的纹理三维模型，需要使用三维扫描设备生成模型。然而却不适用于没有纹理三维模型的领域。之前的工作来处理对称物体：在渲染训练图像时限制位姿的范围，或者对于带注释的真实图像，将范围外的位姿转换为范围内的对称位姿。这种方法适用于单轴对称的物体，只需忽略围绕轴的旋转即可。
<font size=4> &#160; &#160; &#160; &#160;然而BB8中指出当仅有有限数量对称位姿时，很难确定视图边界附近的位姿。文中给的例子是：

> <font size=4>if a box has an angle of symmetry, π, with respect to an axis and a view limit between 0 and π, the pose at π + α(α≈ 0; α > 0) has to be transformed to a symmetric pose at α even if the detailed appearance is closer to a pose at π.

<font size=4> &#160; &#160; &#160; &#160;**如果边界框相对于轴具有对称角π，并且视限在0和π之间，则必须将π+α（α≈0;α> 0）处的位姿转换为α处的对称位姿，即使α足够小，使外观很接近于π位置的位姿。**
pix2pose通过隐式估计被遮挡像素的三维坐标，实现鲁棒性。使用无纹理三维模型从RGB图像回归像素级三维坐标。新的损失函数处理有限个模糊视图的对称物体。

<font size=4>**作者提出目前方法的缺点：**
<font size=4> &#160; &#160; &#160; &#160;1.使用CNN的方法来直接预测投影点的三维边界框，视点以及四元数转换。这些方法都是直接计算的。缺点是缺乏对应关系，这些关系可用于生成多个位姿假设，对被遮挡物体进行鲁棒的估计。对称物体通常通过限制视点范围，这样会增加额外的处理，例如BB8对视图范围进行分类，PoseCNN计算转换后的模型在估计位姿和标注位姿中到最近点的平均距离。然而寻找最近的三维点很耗时。
<font size=4> &#160; &#160; &#160; &#160;2.特征匹配法：AAE只使用RGB图像无监督训练位姿的隐式表示，隐式表示可以接近任何对称的视图，然而使用给定一个好的旋转估计的渲染模板很难指定三维平移。利用二维边界框的大小来计算三维平移的z分量，一旦二维边界框有较小的误差，则影响三维平移。
<font size=4> &#160; &#160; &#160; &#160;3.预测物体空间中像素或局部形状的三维位置。通过对每个像素回归三维坐标并预测类别。这种方法速度较慢。作者使用了一个独立的2D检测网络来提供目标物体的感兴趣区域。
<font size=4> &#160; &#160; &#160; &#160;4.使用自编码的方法生成模型，以对模型去噪、恢复图像缺失的部分。**作者训练了一个带有GAN的自编码网络，可以像图像间转换那样精确的将彩色图像转换为坐标值，像图像绘制那样恢复被遮挡的部分。**

## 创新点

<font size=4>1.提出新的框架：pix2pose，使用无纹理三维模型从RGB回归像素级三维坐标。
<font size=4>2.提出新的损失函数：transformer loss，用于处理有限个模糊视图的对称物体。
<font size=4>3.在LineMOD、LineMOD Occlusion和T-Less上，即使遇到遮挡和对称问题，效果sota。

## 网络结构
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191117192723153.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2xoNjQxNDQ2ODI1,size_16,color_FFFFFF,t_70#pic_center)

<font size=4> &#160; &#160; &#160; &#160;网络输入一张剪裁后的图像Is，输出是物体坐标中每个像素的归一化三维坐标I3D和每个预测的估计误差Ie。Ie将每个像素作为一个置信度，在进行位姿计算之前，直接用来确定内点和外点像素。目标输出包括被遮挡部分的坐标预测，由于坐标由三个值组成，和RGB值相似，因此可以将输出看作彩色图像，从而通过在ground truth位姿中渲染彩色坐标模型，即可得到ground truth 的输出，如下图所示。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191117192741234.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2xoNjQxNDQ2ODI1,size_16,color_FFFFFF,t_70#pic_center)
<font size=4> &#160; &#160; &#160; &#160;即将每个顶点的归一化坐标直接映射到颜色空间中的红绿蓝值，建立每个像素的2D-3D对应，不需要特征匹配。
<font size=4> &#160; &#160; &#160; &#160;卷积核的大小和前四个卷积层与AAE中是相同的。

> <font size=4>To maintain details of low-level feature maps, skip connections are added by copying the half channels of outputs from the first three layers to the corresponding symmetric layers in the decoder, which results in more precise estimation of pixels around geometrical boundaries.

<font size=4> &#160; &#160; &#160; &#160;**为了保证底层特征图的细节，将前三层输出的半通道复制到解码器对应的对称层来添加残差连接。使边界附近的像素估计更准确。每个卷积层与反卷积层的卷积核大小固定为5×5，在编码器与解码器之间有两个全连接层，除了最后一层，所有中间层的输出都采用归一化批处理和激活函数leakyReLU。在最后一层，三通道的输出和激活函数tanh生成三维坐标图像，单通道的输出和激活函数sigmoid估计期望误差Ie。**

## 损失函数

<font size=4> &#160; &#160; &#160; &#160;在三维坐标回归重建目标图像时，使用每个像素的平均L1距离，由于物体的像素比背景更重要，物体mask的误差由mask中的权重误差乘  β因子。basic reconstruction loss：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191117192550552.png#pic_center)
<font size=4> &#160; &#160; &#160; &#160;其中n是像素的数量，Igt表示目标图像的第i个像素，M是目标图像中完全可见物体的mask，mask也包含遮挡部分，用于预测被遮挡物体的不可见部分的值，从而实现被遮挡物体的鲁棒估计。
<font size=4> &#160; &#160; &#160; &#160;上述loss不能处理对称物体，因为其惩罚3维空间中距离较大的像素，没有对称的先验知识。将三维变换矩阵乘上目标图像，可以将每个像素的三维坐标转换成对称位姿。使用候选对称位姿中误差最小的位姿来计算损失函数，transformer loss：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191117192600524.png#pic_center)
<font size=4> &#160; &#160; &#160; &#160;其中Rp是一个位姿到对称位姿的转换，这种损失函数适用于具有有限个对称位姿的目标物体。
<font size=4>transformer loss的效果如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191117192809339.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2xoNjQxNDQ2ODI1,size_16,color_FFFFFF,t_70#pic_center)
<font size=4> &#160; &#160; &#160; &#160;图中可以看出L1 loss在π附近产生了较大的errors；而transformer loss在0到π范围内产生最小值，预计为obj-05的对称角为π。
<font size=4>预测误差计算预测图像与目标图像之间的差异，error prediction loss：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191117192512812.png#pic_center)
<font size=4> &#160; &#160; &#160; &#160;GAN网络能够使用另一个领域的图像生成目标领域中更精确真实的图像，论文中将RGB图像转换为3维坐标图，可以使用GAN网络实现。鉴别器网络能够分辨3维坐标图像是由模型渲染的还是估计的。GAN网络的损失函数为：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191117192454171.png#pic_center)
<font size=4>其中D为鉴别网络。
<font size=4> &#160; &#160; &#160; &#160;**总的损失函数为：**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191117192438370.png#pic_center)
<font size=4> 其中λ1和λ2用于平衡不同的任务。文中后面提到λ1=100，λ2=50.

## 位姿预测过程

<font size=4> &#160; &#160; &#160; &#160;首先使用每个边界框的中心、宽高来剪裁感兴趣区域，调整输入大小为128×128px，然后**将他们乘以1.5，防止剪裁区域包括被遮挡部分**。论文中的位姿预测分为两个阶段，如下图所示：

<font size=4> &#160; &#160; &#160; &#160;第一阶段：由于2维目标检测方法不同，可能导致物体偏移，将边界框与物体中心对齐，消除背景和不确定像素。

> <font size=4>In this stage, the predicted coordinate image I3D is used for specifying pixels that belong to the object including the occluded parts by taking pixels with non-zero values.

<font size=4> &#160; &#160; &#160; &#160;所预测的坐标图像I3D通过取非零值的像素来指定包括遮挡部分的物体像素。如果像素的误差预测大于外点阈值θo，则使用预测误差来移除不确定的像素。物体有效的mask通过非零值的像素单元和误差小于θo的像素来计算。边界框的新中心由有效mask的形心来确定。第一阶段的输出是细化的输入，它只包含从新边界框中裁剪的有效mask的像素。当误差预测小于外点阈值θo时，细化的输入可能包含遮挡部分，这就意味着尽管有遮挡，这些像素的坐标依然很容易预测。**外点阈值θo由三个值决定，目的是包括更多的可见像素，使用人工遮挡的训练图来去除噪点像素。**
<font size=4> &#160; &#160; &#160; &#160;第二阶段：使用第一阶段细化后的图像预测最终的位姿和期望误差。当预测误差大于内点阈值θi时，三维坐标样本中的黑色像素表示点被移除，即使点有非零坐标值。换句话说，具有非零坐标值且误差预测小于阈值θi的像素用来构建2D-3D的对应关系。图像中每个像素已经具有物体坐标中三维点的值，所以二维图像坐标和预测三维坐标直接形成对应。之后利用带有RANSAC的PnP算法，通过最大化内点数量迭代计算最终位姿，内点的二维投影误差比阈值θre更小。
<font size=4> &#160; &#160; &#160; &#160;pix2pose对T-LESS数据集进行评估优点显著，因为T-LESS提供了无纹理CAD模型，而且大多数物体都是对称的，在工业领域更常见。作者将物体的图像从真实图像中提取出来粘贴到COCO数据集中，对图像进行颜色增强处理后，将物体与背景之间的边界进行模糊处理，使边界平滑。用背景图像代替物体区域的一部分来模拟遮挡，最后对增强后的彩色图像和目标坐标图像进行随机旋转。使用Resnet-101的Faster R-CNN和Resnet-50的Retinanet来预测检测到的物体类别，使用COCO数据集的预训练权重对网络进行初始化。

## 度量指标

<font size=4> &#160; &#160; &#160; &#160;对于LineMOD数据集，使用AD{D|I}进行度量，测量ground truth位姿与估计位姿之间顶点的平均距离，对称物体则使用到最近顶点的平均距离，当误差小于物体最大三维直径的1/10时，位姿是正确的。
<font size=4> &#160; &#160; &#160; &#160;对于T-LESS数据集，使用可视表面偏差(VSD)作为度量，只度量可见部分的距离误差，使得度量不受对称和遮挡的影响。

## 实验和结果

<font size=4> 在LineMOD数据集上的结果如下图：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191117192857963.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2xoNjQxNDQ2ODI1,size_16,color_FFFFFF,t_70#pic_center)
<font size=4> 在不使用细化的方法中，作者的方法处理对称物体的效果最好。
<font size=4>在LineMOD Occlusion数据集上的结果如下图：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191117192918710.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2xoNjQxNDQ2ODI1,size_16,color_FFFFFF,t_70#pic_center)
<font size=4> pix2pose的效果明显优于yolo-6d，在八种物体中有三种的效果sota。
<font size=4> 在T-Less数据集上的结果如下图：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019111719293778.png#pic_center)
<font size=4>作者的方法优于现有的使用RGB和RGB-D的方法。
<font size=4> &#160; &#160; &#160; &#160;**文中没有对比PVNet**，可能是因为ICCV是三月截稿，作者没来得及对比。
<font size=4>PVNet在遮挡数据集上的ADD(-S)实验结果为：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191117192949524.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2xoNjQxNDQ2ODI1,size_16,color_FFFFFF,t_70#pic_center)
<font size=4> &#160; &#160; &#160; &#160;可以看出，在遮挡数据集上，pix2pose算法的ADD(-S)实验结果在平均准确率上低于PVNet的实验结果，目标物体eggbox的实验结果差距较大。

<font size=4> &#160; &#160; &#160; &#160;最近比较忙，所以更新可能稍微慢一些，目前正在测试这篇论文的代码，还有一部分关于Ablation studies的，过几天补更。感谢关注！

