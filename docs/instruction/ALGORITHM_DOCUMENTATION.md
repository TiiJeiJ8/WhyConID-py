# WhyConID 算法文档

本文档包含项目中核心算法的理论介绍、数学推导、实现细节与代码示例，方便将来用 Python 或其它语言复现与优化。

---

**目录**
- 一、问题概述
- 二、图像预处理（理论与实践）
- 三、圆/椭圆检测与拟合（理论、推导、实现）
  - 3.1 基于轮廓的检测流程
  - 3.2 几何最小二乘圆拟合（代数方法）
  - 3.3 Taubin/Pratt 拟合要点
  - 3.4 椭圆拟合与投影畸变处理
- 四、偏心 / 离轴圆检测（原理与校正）
- 五、Necklace 风格 ID 生成（理论与实现）
  - 5.1 极角排序与码字生成
  - 5.2 旋转不变与最小表示法
- 六、LGMDProcessing（处理流水线与时序滤波简介）
- 七、参数与鲁棒性建议
- 八、复杂度分析
- 九、Python 关键函数示例（可直接复用或作为模板）
- 十、测试与验证建议

---

## 一、问题概述
WhyConID 的目标是从图像帧中检测圆形或环状标记（可能为亮点或孔的集合），精确求取其几何参数（中心、半径、椭圆轴与偏角），并基于环上特征点序列生成稳定的 ID（对旋转/部分遮挡具有鲁棒性）。这要求组合经典的图像处理、几何拟合与编码策略。

## 二、图像预处理（理论与实践）
输入图像的预处理决定后续检测质量。常见步骤：
- 灰度化：$I_g=0.299R+0.587G+0.114B$。
- 去噪：高斯平滑 $G_{\sigma}$，降低边缘噪声。
- 对比增强：CLAHE 或直方图均衡化改善局部对比度。
- 边缘/阈值：使用自适应阈值或 Canny 边缘检测为轮廓提取提供稳定输入。

实践建议：对不同光照条件使用 CLAHE，再结合自适应阈值；在极端噪声下先用中值滤波。

## 三、圆/椭圆检测与拟合
### 3.1 基于轮廓的检测流程
步骤：
1. 预处理得到二值或边缘图。
2. 使用 `findContours` 提取轮廓集合。
3. 过滤：按 `contourArea`、周长、形状紧致度（compactness）等筛除噪声轮廓。紧致度可定义为 $Q = 4\pi A / P^2$，理想圆的 $Q\approx1$。
4. 使用 `minEnclosingCircle` 或 `fitEllipse` 得到初始参数。
5. 对候选使用几何/亮度一致性验证（环上点亮度分布、内外环差值、边缘强度平均值）。

### 3.2 几何最小二乘圆拟合（代数方法）
目标：给定 $N$ 个点 $(x_i,y_i)$，拟合圆 $(x-a)^2+(y-b)^2=r^2$。
令 $f(x,y)=(x-a)^2+(y-b)^2-r^2=0$，可展开为代数形式：
$$x^2+y^2-2ax-2by+(a^2+b^2-r^2)=0.$$ 令 $D=-2a, E=-2b, F=a^2+b^2-r^2$，则
$$x^2+y^2+Dx+Ey+F=0.$$
代数最小二乘通过求解线性系统：令
$$A=\begin{bmatrix} x_1 & y_1 & 1 \\ \vdots & \vdots & \vdots \\ x_N & y_N & 1 \end{bmatrix},\quad b=-(x_i^2+y_i^2).$$
解 $[D,E,F]^T = (A^TA)^{-1}A^Tb$。
然后 $a=-D/2,\; b=-E/2,\; r=\sqrt{a^2+b^2-F}$。

注意：代数最小二乘对噪声与离群点敏感，需在轮廓点上做子采样并结合 RANSAC 或加权拟合。

### 3.3 Taubin/Pratt 拟合要点
Taubin/Pratt 等代数几何拟合通过约束避免平凡解并减小偏差，思想是在代数残差上做约束最小化，通常需要求解广义特征值问题。实现复杂度高但数值更稳定。若需要高精度，建议使用现成实现或 `scipy.optimize.least_squares` 进行非线性最小二乘直接拟合圆参数：
最小化目标：
$$\min_{a,b,r}\sum_{i=1}^N (\sqrt{(x_i-a)^2+(y_i-b)^2}-r)^2.$$
用 LM（Levenberg–Marquardt）可得高精度参数。

### 3.4 椭圆拟合与投影畸变处理
当物体是圆但成像为椭圆（视角投影）时，使用椭圆拟合（OpenCV 的 `fitEllipse`）得到中心 $(x_0,y_0)$、长短轴 $(A,B)$ 与旋转角 $\theta$。若已知相机内参与无畸变前提下，可尝试从观测椭圆恢复原始圆半径与平面朝向（需要相机标定数据）。否则可把椭圆当作目标形态，后续基于椭圆参数做角度排序与比对。

## 四、偏心 / 离轴圆检测（原理与校正）
偏心或离轴的场景指圆环中心与亮点分布中心不完全重合。检测要点：
- 估计环的几何中心与 brightness-weighted center（亮度质心），比较两者偏移量。
- 如果偏移显著，先用椭圆拟合恢复长短轴与方向，再在椭圆参数坐标系中做角度排序与位置归一化。
- 多帧融合：在多帧上用卡尔曼滤波或滑动窗口平均中心，提高稳健性。

## 五、Necklace 风格 ID 生成
### 5.1 极角排序与码字生成
流程：
1. 对每个检测到的环，得到其中心 $(a,b)$ 并搜集环上关键点集合 $\{(x_i,y_i)\}$（例如轮廓上的亮点或局部极值点）。
2. 将点坐标转换为极坐标：$\theta_i=\operatorname{atan2}(y_i-b,x_i-a)$，并按 $\theta$ 升序排序。
3. 以排序后的点序列构造码字：例如对每个角段评估是否存在亮点 -> 二进制序列；或记录相邻点间角度间隔、归一化后量化为符号序列。

### 5.2 旋转不变与最小表示法
因为目标可以任意旋转，需对循环序列做循环等价类归一化：找到序列所有循环移位的字典序最小（或最大）表示作为 canonical form。示例：对二进制序列 s，生成所有旋转 s_k，选择最小者为 ID。对抗镜像（左右翻转）可同时比较翻转序列并取最小者。

若允许少量缺点/遮挡，使用鲁棒的匹配策略：计算循环海明距离最小值，若低于阈值则视为匹配。

## 六、LGMDProcessing（处理流水线简介）
LGMD（Lobula Giant Movement Detector）常用于生物视觉启发的运动/边缘响应检测；在本项目中，`LGMDProcessing` 模块可被理解为针对局部对比、时序变化或运动特征的滤波与响应增强模块。实现要点：
- 时序滤波：保持上一帧状态 $S_{t-1}$，当前输入 $I_t$，用高通或差分 $D_t=I_t-I_{t-1}$ 提取变化。
- 空间滤波：使用差分 of Gaussians（DoG）或拉普拉斯增强边缘。
- 响应门控：阈值化 $D_t$ 并结合局部能量评估防止噪声触发。

## 七、参数与鲁棒性建议
- 轮廓最小面积阈值应基于期望最小半径 $r_{min}$，例如 $A_{min}=\pi r_{min}^2/4$（保守）。
- Canny 阈值可采用自动估计：低阈值 = 0.66 * median, 高阈值 = 1.33 * median of gradient magnitude。
- 对拟合使用 RANSAC：在代数拟合前用 RANSAC 选出内点集合，提高对杂点的鲁棒性。
- 对亮度/光照变化使用局部自适应阈值与 CLAHE。

## 八、复杂度分析
- 预处理（滤波、Canny）：O(MN) 对图像像素数。
- 轮廓提取：近 O(K)（K 为边缘像素数），通常 < O(MN)。
- 拟合（最小二乘）：线性系统求解 O(n)（n 为轮廓点数，使用子采样）。
总体实时性受图像分辨率与轮廓复杂度影响。可通过缩放输入或在 ROI 上运行来加速。

## 九、Python 关键函数示例
下面给出可直接使用或作为模板的 Python 实现：

### 9.1 代数圆拟合（线性最小二乘）
```python
import numpy as np

def algebraic_circle_fit(pts):
    # pts: Nx2 array of (x,y)
    X = pts[:,0]
    Y = pts[:,1]
    A = np.column_stack([X, Y, np.ones_like(X)])
    b = -(X**2 + Y**2)
    # Solve for D,E,F
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    D, E, F = sol
    a = -D/2.0
    b_ = -E/2.0
    r = np.sqrt(a*a + b_*b_ - F)
    return (a, b_, r)
```
说明：结果受离群点影响，建议在 RANSAC 或子采样后调用。

### 9.2 非线性最小二乘（直接拟合距离）
```python
from scipy.optimize import least_squares

def residuals_circle(params, x, y):
    a, b, r = params
    return np.sqrt((x-a)**2 + (y-b)**2) - r

def nonlinear_circle_fit(pts):
    x = pts[:,0]; y = pts[:,1]
    x_m = x.mean(); y_m = y.mean()
    r0 = np.mean(np.sqrt((x-x_m)**2 + (y-y_m)**2))
    p0 = [x_m, y_m, r0]
    res = least_squares(residuals_circle, p0, args=(x,y))
    return tuple(res.x)
```

### 9.3 Necklace ID（极角排序 + 最小旋转表示）
```python
import numpy as np

def canonical_rotation_binary(bin_list):
    # bin_list: list/array of 0/1
    s = ''.join(str(int(x)) for x in bin_list)
    rotations = [s[i:]+s[:i] for i in range(len(s))]
    return min(rotations)

def generate_necklace_id(center, points):
    cx, cy = center
    angles = np.arctan2(points[:,1]-cy, points[:,0]-cx)
    order = np.argsort(angles)
    ordered = points[order]
    # Example: threshold by local brightness or presence -> construct binary
    # Here we create simple binary by checking intensity at point if provided
    # For demo, assume points have third column intensity
    bits = (ordered[:,2] > 128).astype(int)
    return canonical_rotation_binary(bits)
```
该实现为示例，实际需要按项目逻辑选用合适的量化/阈值策略。

### 9.4 RANSAC + 拟合模板
对含明显噪声的轮廓，使用 RANSAC：随机采样 3 点拟合圆（解析解），计算内点数，迭代取最佳模型，然后用内点集做精拟合（非线性最小二乘）。

## 十、测试与验证建议
- 从 `bin/Debug` 中拷贝若干带注释的样例图像到 `sample_data/`，并对每张图运行检测，记录中心与半径，与手工标注比较。 
- 对 `Necklace` ID，构造不同旋转/部分遮挡情形验证匹配率
- 写 `pytest` 测试覆盖 `algebraic_circle_fit`、`nonlinear_circle_fit` 与 `generate_necklace_id` 的边界情况。

---

## 附：参考与进一步阅读
- Forsyth, D. A., & Ponce, J. (2002). Computer Vision: A Modern Approach.
- Taubin, G. (1991). Estimation Of Planar Curves, Surfaces And Nonplanar Space Curves Defined By Implicit Equations, With Applications To Edge And Range Image Segmentation.
- OpenCV 文档：`findContours`, `fitEllipse`, `minEnclosingCircle`, `warpPerspective`。

---

文档路径：[WhyConID/ALGORITHM_DOCUMENTATION.md](WhyConID/ALGORITHM_DOCUMENTATION.md)
