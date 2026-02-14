# 单目 3D 定位 — 数学推导与算法说明

本文档以数学为主导，描述单目相机下基于标记尺寸恢复深度与将像素点恢复到世界坐标系的完整推导、误差传播与实现要点。

## 1. 约定与符号

- 像素坐标：$(u,v)$，齐次记作 $\tilde{p}=[u\;v\;1]^T$.
- 相机内参：$K=\begin{bmatrix}f_x & 0 & c_x\\ 0 & f_y & c_y\\ 0 & 0 & 1\end{bmatrix}$（像素单位）。
- 相机在世界坐标系的位姿：位置 $C=[C_x\;C_y\;C_z]^T$（米），旋转矩阵 $R_{cw}$（将世界向量变换到相机坐标）。约定：

  $$
  P_c = R_{cw}(P_w - C).
  $$

  则逆变换为

  $$
  P_w = R_{cw}^T P_c + C.
  $$
- 标记真实直径：$D_{real}$（米），图像中测得像素直径：$D_{px}$（像素）。

## 2. 针孔模型与像素—相机坐标关系

针孔投影：对任意相机坐标点 $P_c=[X_c\;Y_c\;Z_c]^T$，有

$$
s\begin{bmatrix}u\\v\\1\end{bmatrix}=K\begin{bmatrix}X_c\\Y_c\\Z_c\end{bmatrix}.
$$

因此反投影（已知 $Z_c$）:

$$
X_c=\frac{(u-c_x)Z_c}{f_x},\quad Y_c=\frac{(v-c_y)Z_c}{f_y}.
$$

## 3. 基于标记尺寸的深度估计（闭式解）

当标记为平面圆形且近似垂直于光轴（无强烈倾斜）时，针孔模型给出：

$$
\frac{D_{px}}{f}\approx\frac{D_{real}}{Z_c}.
$$

解得深度：

$$
\boxed{\;Z_c = f\cdot\dfrac{D_{real}}{D_{px}}\;}
$$

其中 $f$ 可取 $f_x$（若使用图像横向像素测量）或 $\bar f=(f_x+f_y)/2$。

其中 $f$ 可取 $f_x$（若使用图像横向像素测量）或 $\bar f=(f_x+f_y)/2$。

实现上 $D_{px}$ 可由三种方式估计：

- 最小外接圆直径；
- 椭圆拟合短轴（对倾斜更稳健）；
- 面积法：$D_{px}=\sqrt{4A/\pi}$（对噪声敏感）。

注意：若标记显著倾斜（法向与相机光轴夹角 $\theta$），实际投影尺寸会乘以余弦因子，需做倾斜校正。

## 4. 从像素到世界坐标的完整流程

1. 检测并获得像素中心 $(u,v)$ 与测得直径 $D_{px}$.
2. 计算深度 $Z_c = f D_{real}/D_{px}$.
3. 计算相机坐标：
   $$
   P_c=\begin{bmatrix}X_c\\Y_c\\Z_c\end{bmatrix}=\begin{bmatrix}\dfrac{(u-c_x)Z_c}{f_x}\\[4pt]\dfrac{(v-c_y)Z_c}{f_y}\\[4pt]Z_c\end{bmatrix}.
   $$
4. 转换到世界坐标：
   $$
   P_w = R_{cw}^T P_c + C.
   $$

此时 $P_w$ 给出标记在世界坐标系下的三维位置（米）。

## 5. 射线/地面交点（不依赖标记尺寸的地面投影）

像素 $(u,v)$ 定义一条穿过相机光心的射线。令非归一化方向向量为

$$
d_c = \begin{bmatrix}(u-c_x)/f_x\\(v-c_y)/f_y\\1\end{bmatrix},
$$

单位方向 $\hat d_c=d_c/\|d_c\|$，其在世界系的方向为 $\hat d_w=R_{cw}^T\hat d_c$. 射线参数化：

$$P_w(t)=C+t\hat d_w.$$
若地面为 $Z_w=0$ 面，求 $t$ 使得 $[P_w(t)]_z=0$，得
$$t^*=-\dfrac{C_z}{\hat d_{w,z}}\quad(\hat d_{w,z}<0).$$

交点 $P_{ground}=C+t^*\hat d_w$。

该方法常用于将像素映射到地面平面（例如机器人场景），在无标记尺寸信息下仍然可用。

## 6. 倾斜标记的几何修正（椭圆到圆的逆推，概述）

若标记面与相机存在倾斜，图像中观测为椭圆。设拟合椭圆短轴 $b$ 与长轴 $a$，若标记半径为 $R= D_{real}/2$ 且相机到标记中心距离为 $Z_c$，投影关系近似：

- 垂直方向的投影尺度与 $b$ 相关，短轴更接近圆在视线方向的截面；
- 若能估计平面法向 $n_w$，则投影尺度校正可由 $|n_w\cdot z_c|$（法向与相机光轴夹角余弦）进行。

精确恢复需解平面-圆投影的几何方程，或通过多视几何/标定板估计平面姿态。

## 7. 不确定性传播（误差估计）

对 $Z_c=fD_{real}/D_{px}$ 做一阶近似传播：若 $D_{px},f,D_{real}$ 方差分别为 $\sigma_D^2,\sigma_f^2,\sigma_{D_r}^2$，则

$$
\mathrm{Var}(Z_c)\approx\left(\frac{\partial Z_c}{\partial D_{px}}\right)^2\sigma_D^2+\left(\frac{\partial Z_c}{\partial f}\right)^2\sigma_f^2+\left(\frac{\partial Z_c}{\partial D_{real}}\right)^2\sigma_{D_r}^2,
$$

代入偏导得：

$$
\mathrm{Var}(Z_c)=\left(\frac{fD_{real}}{D_{px}^2}\right)^2\sigma_D^2+\left(\frac{D_{real}}{D_{px}}\right)^2\sigma_f^2+\left(\frac{f}{D_{px}}\right)^2\sigma_{D_r}^2.
$$

相对误差近似为：

$$
\frac{\sigma_Z}{Z_c}\approx\sqrt{\left(\frac{\sigma_D}{D_{px}}\right)^2+\left(\frac{\sigma_f}{f}\right)^2+\left(\frac{\sigma_{D_r}}{D_{real}}\right)^2 }.
$$

实用含义：像素直径的不确定性对深度影响最大（以 $\propto 1/D_{px}$ 或 $\propto 1/D_{px}^2$ 的尺度），当 $D_{px}$ 很小时深度误差迅速升高。

## 8. 联合优化（重投影误差最小化）

当存在多帧观测或多标记几何约束时，优先用非线性最小二乘细化位姿或深度：
$$
\min_{P_w}\sum_i\left\|\pi\big(R_{cw}(P_w-C)\big)-p_i\right\|^2,
$$

其中 $\pi([X\;Y\;Z]^T)=(f_xX/Z+c_x,\;f_yY/Z+c_y)$。可同时把 $D_{real}$、相机外参或 $P_w$ 作为待估量，使用 LM 或 Ceres 求解并输出协方差近似。

## 9. 推荐实践

- 使用标定得到的 $f_x,f_y,c_x,c_y$ 与去畸变图像；优先用焦距 $f_x$（横向测量）或对 $f_x,f_y$ 做适当组合。
- 精确测量 $D_{real}$（米），并统一单位。
- 对广角镜头先去畸变或使用 `--crop-border` 避免边缘畸变影响。
- 对倾斜标记采用椭圆短轴或进行平面姿态估计以校正尺度。
- 在跟踪中使用滤波器（卡尔曼/扩展卡尔曼）融合多帧深度估计以降低噪声。

## 10. 伪代码流程

```
for each frame:
    preprocess(frame)
    detect mark -> (u,v), estimate D_px (circle/ellipse/area)
    if enable_depth:
        Z = f * D_real / D_px
        X = (u - c_x) * Z / f_x
        Y = (v - c_y) * Z / f_y
        P_c = [X,Y,Z]
        P_w = R_cw^T * P_c + C
    else:
        # optional: compute ground intersection from ray
        compute direction d_c = [(u-c_x)/f_x, (v-c_y)/f_y, 1]
        d_w = R_cw^T * normalize(d_c)
        intersect with Z_w=0 to get ground point
    append to tracker / visualize
```

---

参考：针孔相机模型、复投影误差优化、椭圆-圆投影几何。

文档位置：`docs/instruction/3D_LOCALIZATION.md`
