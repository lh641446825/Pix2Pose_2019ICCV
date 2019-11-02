# Pix2Pose:Pixel-Wise Coordinate Regression of Objects for 6D Pose Estimation

文中提到之前处理对称物体的方法是在渲染训练图时限制位姿的范围，或者对于带注释的真实图像，将有限范围外的位姿转换为范围内的对称位姿。这种方法对于单轴无限或连续对称的物体是足够的，只需忽略围绕轴的旋转即可。然而BB8中指出当仅有有限数量对称位姿时，很难确定视图边界附近的位姿。文中给的例子是：
if a box has an angle of symmetry, π, with respect to an axis and a view limit between 0 and π, the pose at π + α(α≈ 0; α > 0) has to be transformed to a symmetric pose at α even if the detailed appearance is closer to a pose at π.

如果边界框相对于轴具有对称角π，并且视限在0和π之间，则必须将π+α（α≈0;α> 0）处的位姿转换为α处的对称位姿，即使α足够小，使外观很接近于π位置的位姿。
