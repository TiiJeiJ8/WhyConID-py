# WhyConID-py 使用文档

WhyConID-py 是一个基于Python的圆形标记检测和识别系统，使用内外环算法检测 WhyConID 标记。

## 核心功能

✨ **标记检测**：基于内外环算法的高精度圆形标记检测  
🎯 **轨迹追踪**：Kalman滤波器多目标追踪，自动ID关联  
🔮 **运动预测**：多步轨迹预测与误差可视化  
📐 **3D定位**：单目深度估计，世界坐标系转换  
📊 **3D可视化**：交互式3D轨迹图（Plotly HTML）  
🎨 **预处理增强**：时序平滑、CLAHE对比度增强  
🔧 **灵活调参**：丰富的命令行参数，适配各种场景

## 快速开始

### 安装依赖

```bash
cd core
pip install -r requirements.txt
```

### 基本使用

```bash
# 检测单张图片
python main.py input.jpg

# 检测并显示结果
python main.py input.jpg --show

# 处理视频文件
python main.py video.mp4 --show

# 使用摄像头实时检测
python main.py 0 --show
```

### 高级功能快速示例

```bash
# 轨迹追踪 + 预测可视化
python main.py video.mp4 --track --show-prediction --prediction-steps 10 --color-trajectory --output tracked.mp4

# 3D定位追踪（基础）
python main.py video.mp4 --track --enable-depth --marker-diameter 50 --fov 60 --export-3d --output 3d.mp4

# 3D定位追踪（完整外参）
python main.py video.mp4 \
    --track --enable-depth --marker-diameter 50 --fov 60 \
    --camera-position 0.3 0.2 1.5 --camera-rotation 0 45 0 \
    --export-3d --save-trajectory --output 3d_full.mp4

# 预处理增强 + 轨迹调参
python main.py video.mp4 \
    --track --temporal-smooth 5 --use-clahe \
    --match-threshold 300 --max-age 120 \
    --persistent-trajectory --output enhanced.mp4
```

## 命令行参数

### 必需参数

- `input`: 输入源
  - 图片文件路径：`image.jpg`, `photo.png`
  - 视频文件路径：`video.mp4`, `video.avi`
  - 摄像头索引：`0` (默认摄像头), `1` (第二个摄像头)

### 可选参数

#### 显示和输出

- `--show`: 实时显示检测结果窗口

  - 按 `q` 或 `ESC` 键退出
- `--output <path>`: 保存处理后的视频文件

  ```bash
  python main.py input.mp4 --output result.mp4
  ```
- `--output-dir <dir>`: 指定输出目录（默认：`output`）

  ```bash
  python main.py input.jpg --save-img --output-dir my_results
  ```

#### 保存选项

- `--save-img`: 保存标注后的图片

  - 输出文件：`frame_XXXX_detected.jpg`
  - 包含标记位置、ID、边界框等可视化信息
- `--save-log`: 保存检测日志

  - `detection_log.txt`: 完整控制台日志
  - `detection_results.txt`: 详细检测结果报告
- `--save-csv`: 导出CSV格式结果

  - `detection_results.csv`: 表格形式的检测数据
  - 包含坐标、面积、圆度等参数

#### 检测参数

- `--markers <N>`: 要跟踪的标记数量（默认：1）

  ```bash
  python main.py input.jpg --markers 5
  ```
- `--debug <level>`: 调试级别（0-3）

  - `0`: 无调试输出（默认）
  - `1`: 基本检测信息
  - `2`: 每个轮廓的详细信息
  - `3`: 完整的调试输出

  ```bash
  python main.py input.jpg --debug 2
  ```
- `--config <path>`: 指定配置文件路径

  ```bash
  python main.py input.jpg --config custom_config.yaml
  ```

#### 轨迹追踪选项

- `--track`: 启用轨迹追踪和预测

  - 使用Kalman滤波器进行位置预测
  - 自动关联多帧之间的标记
  - 为每个标记分配唯一ID（T0, T1, T2...）
- `--save-trajectory`: 保存轨迹数据到CSV

  - `trajectories.csv`: 完整的轨迹历史
  - 包含Track ID、帧号、位置、预测位置、速度等
- `--persistent-trajectory`: 持久化轨迹显示

  - 保持完整轨迹历史（不限制长度）
  - 默认模式只显示最近50个点
  - 适合分析完整运动路径
- `--color-trajectory`: 多色轨迹显示

  - 每条轨迹使用不同颜色
  - 便于区分多个目标
  - 颜色自动从预定义调色板选择

#### 轨迹预测选项

- `--show-prediction`: 显示轨迹预测
  - 方案A：单步箭头（当前位置→下一步预测）
  - 方案B：多步虚线轨迹（半透明彩色路径）
- `--prediction-steps <N>`: 预测未来N步（默认：5）
  ```bash
  python main.py video.mp4 --track --show-prediction --prediction-steps 10
  ```
- `--show-prediction-error`: 显示预测误差
  - 红色X标记上一帧的预测位置
  - 绿色线连接预测与实际位置
  - 显示像素误差距离

#### 预处理选项 🎨

- `--temporal-smooth <帧数>`: 时序平滑处理（默认：3帧）
  - 对N帧进行平均，减少噪声
  - 数值越大越平滑，但延迟增加
  - 适合光照不稳定的场景
  
- `--use-clahe`: 启用对比度受限自适应直方图均衡化
  - 自动增强局部对比度
  - 改善低对比度或光照不均场景的检测
  - 配合 `--clahe-clip` 和 `--clahe-grid` 调节效果
  
- `--clahe-clip <值>`: CLAHE裁剪限制（默认：2.0）
  - 控制对比度增强强度
  - 数值增大增强效果更明显
  
- `--clahe-grid <大小>`: CLAHE网格大小（默认：8）
  - 局部区域划分的网格尺寸
  - 增大可处理更大的光照变化区域
  
- `--crop-border <像素>`: 裁剪图像边界（默认：0）
  - 移除边缘N像素，避免边缘畸变影响检测
  - 广角镜头或鱼眼相机建议使用（10-50像素）

**预处理示例：**

```bash
# 光照不稳定场景
python main.py video.mp4 --temporal-smooth 5 --use-clahe --track --output smooth.mp4

# 广角相机 + 对比度增强
python main.py video.mp4 --crop-border 30 --use-clahe --clahe-clip 3.0 --track --output enhanced.mp4
```

#### 跟踪器调参选项 🔧

- `--match-threshold <距离>`: 匹配距离阈值（默认：200.0像素）
  - 两帧间标记中心距离超过此值视为不同目标
  - 快速运动场景建议增大（300-500）
  - 密集场景建议减小（50-150）
  
- `--max-age <帧数>`: 丢失后保留帧数（默认：90帧）
  - 标记消失后继续保留ID的最大帧数
  - 遮挡频繁场景建议增大（120-300）
  - 实时性要求高可减小（30-60）
  
- `--min-hits <次数>`: 确认跟踪的最小命中数（默认：8次）
  - 新目标需连续检测多少次才确认为有效轨迹
  - 噪声多时增大（10-15）减少误跟踪
  - 快速响应场景减小（3-5）
  
- `--memory-frames <帧数>`: ID恢复记忆帧数（默认：500帧）
  - 记住已丢失轨迹的时长，用于ID重识别
  - 长时间遮挡场景增大（1000-3000）
  - 短暂检测场景减小（100-300）

#### 3D定位与深度估计 📐

##### 深度估计基础参数

- `--enable-depth`: 启用深度估计（从标记尺寸计算距离）
  - 基于针孔相机模型：Z = f × D_real / D_pixel
  - 需要提供标记真实尺寸和相机焦距或视场角
  
- `--marker-diameter <直径>`: 标记真实直径（单位：毫米，默认：50.0）
  - 必须精确测量实际标记的直径
  - 深度估计精度直接依赖此参数
  
- `--focal-length <像素>`: 相机焦距（单位：像素）
  - 通过相机标定获得
  - 与 `--fov` 二选一（优先使用焦距）
  
- `--fov <角度>`: 水平视场角（单位：度）
  - 与 `--focal-length` 二选一
  - 典型值：普通镜头60°，广角90-120°
  - 自动计算焦距：f = width / (2 × tan(FOV/2))

##### 相机外参配置（世界坐标系）

- `--camera-position X Y Z`: 相机在世界坐标系中的位置（单位：米）
  - X: 左右偏移（正值向右）
  - Y: 前后偏移（正值向前）
  - Z: 垂直高度（正值向上）
  - 示例：`--camera-position 0.3 0.2 1.5` 表示相机在右侧30cm、前方20cm、高度1.5m处
  
- `--camera-rotation ROLL PITCH YAW`: 相机旋转角度（单位：度）
  - ROLL: 绕光轴旋转（翻滚）
  - PITCH: 俯仰角（正值向下倾斜）
  - YAW: 偏航角（正值向右旋转）
  - 示例：`--camera-rotation 0 45 0` 表示相机向下倾斜45度

##### 3D可视化导出

- `--export-3d`: 导出3D轨迹可视化
  - 生成交互式HTML文件（可在浏览器中旋转/缩放）
  - 生成高清静态PNG图像（用于论文/报告）
  - 需要同时启用 `--enable-depth` 和 `--track`

**3D定位示例：**

```bash
# 基础深度估计（相机默认正上方）
python main.py video.mp4 --track --enable-depth --marker-diameter 50 --fov 60 --output depth.mp4

# 完整3D追踪（指定相机位置和角度）
python main.py video.mp4 \
    --track --enable-depth \
    --marker-diameter 50 --fov 60 \
    --camera-position 0.3 0.2 1.5 \
    --camera-rotation 0 45 0 \
    --export-3d \
    --output 3d_tracked.mp4

# 高精度3D定位（使用标定焦距）
python main.py video.mp4 \
    --track --enable-depth \
    --marker-diameter 45.5 --focal-length 850.2 \
    --camera-position 0 0 2.0 --camera-rotation 0 30 0 \
    --export-3d --save-trajectory \
    --output calibrated_3d.mp4

# 仅生成3D可视化（不保存视频）
python main.py video.mp4 \
    --track --enable-depth --marker-diameter 50 \
    --camera-position 0 0 1.8 --camera-rotation 0 40 0 \
    --export-3d --save-trajectory
```

**坐标系定义：**

- **相机坐标系**：X右，Y下，Z前（沿光轴）
- **世界坐标系**：X右，Y前，Z上（右手系）
- 相机位置和旋转参数定义相机→世界的变换关系

**如何测量相机外参：**

1. **位置（X, Y, Z）**：
   - 在场景中选择一个原点（如检测区域中心）
   - 测量相机镜头中心相对原点的三维坐标
   - X轴向右为正，Y轴向前为正，Z轴向上为正

2. **旋转（Roll, Pitch, Yaw）**：
   - Roll（翻滚）：通常为0（除非相机倾斜安装）
   - Pitch（俯仰）：相机向下看时为正值（如45°）
   - Yaw（偏航）：相机向右偏时为正值（通常为0）

**轨迹追踪示例：**

```bash
# 基础轨迹追踪
python main.py video.mp4 --track --show --markers 2

# 保存轨迹数据
python main.py video.mp4 --track --save-trajectory --markers 3

# 持久化彩色轨迹 + 视频导出
python main.py video.mp4 --track --persistent-trajectory --color-trajectory --output tracked.mp4

# 带预测可视化
python main.py video.mp4 --track --show-prediction --prediction-steps 10 --color-trajectory --output predicted.mp4

# 预测误差分析
python main.py video.mp4 --track --show-prediction --show-prediction-error --persistent-trajectory --output error_analysis.mp4

# 完整功能演示
python main.py video.mp4 --track --save-trajectory --persistent-trajectory --color-trajectory --show-prediction --prediction-steps 10 --output result.mp4 --markers 5
```

**调参优化示例：**

```bash
# 快速运动场景（放宽匹配、延长记忆）
python main.py fast_motion.mp4 --track --match-threshold 300 --max-age 120 --memory-frames 1000 --output fast.mp4

# 密集场景（严格匹配、快速确认）
python main.py dense.mp4 --track --match-threshold 80 --min-hits 5 --max-age 60 --markers 10 --output dense.mp4
```

## 使用示例

### 1. 基础检测

```bash
# 检测图片并显示
python main.py TEST/img_test.jpg --show

# 检测视频并保存结果
python main.py video.mp4 --output result.mp4
```

### 2. 完整输出

```bash
# 保存所有结果（图片、日志、CSV）
python main.py input.jpg --save-img --save-log --save-csv

# 检测多个标记并保存
python main.py input.jpg --markers 10 --save-img --save-csv
```

### 3. 调试模式

```bash
# 显示详细检测过程
python main.py input.jpg --show --debug 2

# 完整调试输出
python main.py input.jpg --debug 3 --save-log
```

### 4. 实时检测

```bash
# 使用摄像头实时检测
python main.py 0 --show --markers 5

# 摄像头检测并录制视频
python main.py 0 --show --output live_capture.mp4
```

### 5. 3D定位与追踪

#### 步骤1：测量标记尺寸
```bash
# 使用卡尺精确测量标记外环直径（毫米）
# 示例：测得直径为 50.5mm
```

#### 步骤2：获取相机参数
```bash
# 方法1：查看相机规格（视场角）
# 例如：某摄像头水平FOV = 60°

# 方法2：相机标定（推荐，精度更高）
# 使用OpenCV标定工具获取焦距
# 例如：标定得 focal_length = 850.2 像素
```

#### 步骤3：测量相机外参
```bash
# 在场景中定义坐标原点（如检测区域中心）
# 测量相机镜头中心相对原点的位置：
#   X = 0.3m (向右)
#   Y = 0.2m (向前) 
#   Z = 1.5m (高度)
# 
# 测量相机朝向：
#   Roll = 0° (无翻滚)
#   Pitch = 45° (向下倾斜45度)
#   Yaw = 0° (无偏航)
```

#### 步骤4：运行3D追踪
```bash
# 使用视场角（方法1）
python main.py video.mp4 \
    --track --enable-depth \
    --marker-diameter 50.5 --fov 60 \
    --camera-position 0.3 0.2 1.5 \
    --camera-rotation 0 45 0 \
    --export-3d --save-trajectory \
    --persistent-trajectory --color-trajectory \
    --output 3d_tracked.mp4

# 使用标定焦距（方法2，精度更高）
python main.py video.mp4 \
    --track --enable-depth \
    --marker-diameter 50.5 --focal-length 850.2 \
    --camera-position 0.3 0.2 1.5 \
    --camera-rotation 0 45 0 \
    --export-3d --save-trajectory \
    --output 3d_calibrated.mp4
```

#### 步骤5：查看3D可视化
```bash
# 打开交互式3D图（浏览器）
# output/run_YYYYMMDD_HHMMSS/trajectory_3d.html

# 查看静态3D图
# output/run_YYYYMMDD_HHMMSS/trajectory_3d.png

# 查看轨迹数据（CSV）
# output/run_YYYYMMDD_HHMMSS/trajectories.csv
```

## 输出文件结构

每次运行会在输出目录下创建独立的时间戳文件夹：

```
output/
└── run_20260206_154219/
    ├── run_summary.txt              # 运行摘要
    ├── detection_log.txt            # 控制台日志（--save-log）
    ├── detection_results.txt        # 详细检测结果（--save-log）
    ├── detection_results.csv        # CSV导出（--save-csv）
    ├── trajectories.csv             # 轨迹数据（--save-trajectory）
    ├── trajectory_3d.html           # 交互式3D可视化（--export-3d）
    ├── trajectory_3d.png            # 静态3D可视化（--export-3d）
    ├── tracked_video.mp4            # 处理后视频（--output）
    └── frame_0001_detected.jpg      # 标注图片（--save-img）
```

### 输出文件说明

#### run_summary.txt

运行概要信息：

- 运行时间
- 输入源
- 分辨率和帧数
- 检测到的标记数量
- 生成的文件列表

#### detection_results.txt

详细的检测报告：

```
Marker #1:
  ID: 0
  Position: (1238.99, 213.99)
  Bounding Box: (1207, 189) -> (1272, 240)
  Size (pixels): 2390
  Roundness: 0.8378
  BW Ratio: 0.2381
  ...
```

#### detection_results.csv

表格格式数据：

```csv
Marker_ID,Center_X,Center_Y,BBox_MinX,BBox_MinY,BBox_MaxX,BBox_MaxY,Size,Roundness,BW_Ratio,M0,M1,Valid
0,1238.99,213.99,1207,189,1272,240,2390,0.8378,0.2381,17.13,11.11,True
```

#### trajectories.csv (轨迹追踪)

轨迹历史数据（使用 `--save-trajectory`）：

```csv
Track_ID,Frame,Timestamp,X,Y,Predicted_X,Predicted_Y,Velocity_X,Velocity_Y
0,1,0.033,1238.99,213.99,1238.99,213.99,0.00,0.00
0,2,0.067,1240.15,215.32,1239.50,214.25,34.82,39.91
0,3,0.100,1242.08,217.45,1241.56,216.89,58.01,63.84
1,1,0.033,856.23,412.67,856.23,412.67,0.00,0.00
1,2,0.067,854.89,413.12,856.01,412.82,-40.21,13.50
```

**字段说明：**

- `Track_ID`: 轨迹唯一标识符
- `Frame`: 帧号
- `Timestamp`: 时间戳（秒）
- `X, Y`: 实际检测位置
- `Predicted_X, Predicted_Y`: Kalman滤波器预测位置
- `Velocity_X, Velocity_Y`: 速度（像素/秒）

#### trajectory_3d.html (交互式3D可视化)

使用 Plotly 生成的交互式3D轨迹图（使用 `--export-3d`）：

**特性：**
- 可在浏览器中打开，支持旋转/缩放/平移
- 显示世界坐标系中的完整轨迹
- 不同轨迹使用不同颜色
- 绿色圆点标记起点，红色方块标记终点
- 包含地平面网格（Z=0）

**使用方法：**
1. 用浏览器打开 `trajectory_3d.html`
2. 鼠标拖动旋转视角
3. 滚轮缩放
4. 右键拖动平移

#### trajectory_3d.png (静态3D可视化)

高分辨率静态3D轨迹图（300 DPI，适合论文使用）：

**特性：**
- 预设视角生成的高质量图像
- 包含坐标轴标签和网格
- 可直接插入研究报告或论文

#### 标注图片/视频

图片/视频上的可视化标记：

- **绿色十字**：标记中心点
- **蓝色矩形框**：边界框
- **黄色文字**：ID（Track ID或Marker ID）、坐标、圆度、面积
- **深度信息**（启用 `--enable-depth` 时）：显示距离（米）和3D坐标
- **橙色轨迹线**（启用追踪时）：标记运动轨迹
- **彩色轨迹**（启用 `--color-trajectory`）：每条轨迹不同颜色
- **空心圆圈**（跟踪丢失时）：标记当前未检测到，正在预测位置
- **预测箭头**（启用 `--show-prediction`）：从当前位置指向下一步预测
- **虚线轨迹**（启用 `--show-prediction`）：显示未来N步的预测路径
- **误差标记**（启用 `--show-prediction-error`）：红色X和绿色线显示预测误差
- **白色面板**：检测统计和时间戳
- **进度条**（视频播放时）：显示播放进度和时间

## 检测算法说明

本系统使用 WhyConID 内外环检测算法：

1. **图像二值化**：自动阈值分割
2. **轮廓检测**：查找黑色外环
3. **内环验证**：验证白色内环的存在
4. **特征计算**：
   - 圆度测试（4πA/P²）
   - 面积比率验证（内环/外环 ≈ 4.95）
   - 同心度检测
   - 长宽比测试
5. **ID识别**：通过 Necklace 编码解码

## 3D定位原理

### 单目深度估计

基于针孔相机模型，从标记在图像中的尺寸估计其距离：

**公式：** `Z = f × D_real / D_pixel`

- `Z`: 标记到相机的距离（深度）
- `f`: 相机焦距（像素单位）
- `D_real`: 标记真实直径（毫米）
- `D_pixel`: 标记在图像中的直径（像素）

**焦距计算（如果只知道FOV）：**

`f = image_width / (2 × tan(FOV_horizontal / 2))`

**精度影响因素：**
1. 标记直径测量精度（±1mm误差在1m距离可造成±2%误差）
2. 相机焦距标定精度
3. 图像分辨率（高分辨率下像素直径更准确）
4. 镜头畸变（广角镜头建议使用 `--crop-border`）

### 坐标系变换

系统支持两种坐标系：

#### 1. 相机坐标系（Camera Frame）
- 原点：相机光学中心
- X轴：向右（图像坐标系）
- Y轴：向下（图像坐标系）
- Z轴：沿光轴向前（深度方向）

#### 2. 世界坐标系（World Frame）  
- 原点：用户定义（通常为检测区域中心）
- X轴：向右
- Y轴：向前（水平面）
- Z轴：向上（垂直方向）

#### 坐标转换

通过相机外参（位置 + 旋转）进行坐标系变换：

**旋转矩阵**（Euler角，ZYX顺序）：
```
R = R_z(yaw) × R_y(pitch) × R_x(roll)
```

**相机→世界：**
```
P_world = R^T × (P_camera - T_camera)
```

**世界→相机：**
```
P_camera = R × P_world + T_camera
```

**地面投影**（假设地面为Z=0平面）：
```
已知相机坐标 (x_c, y_c, z_c) 和相机外参
求解射线与地面交点，得到地面坐标 (x_w, y_w, 0)
```

### 应用场景

1. **机器人导航**：将检测到的标记转换为机器人世界坐标
2. **多相机融合**：统一不同相机视角下的目标位置
3. **运动分析**：在真实物理坐标系中分析运动轨迹
4. **高度测量**：计算目标相对地面的高度

## 性能建议

### 图片处理

- 推荐分辨率：640x480 到 1920x1080
- 标记直径：至少 20 像素

### 视频处理

- 30 FPS 以下可实时处理
- 高分辨率视频建议降低帧率

### 摄像头

- 建议使用 640x480 或 1280x720 分辨率
- 确保光照充足，标记清晰可见

## 常见问题

### 检测不到标记？

1. 确保标记清晰，对比度高
2. 调整相机角度，避免严重变形
3. 增加调试级别查看详细信息：`--debug 2`
4. 检查标记是否符合 WhyConID 规范（黑色外环+白色内环）

### 误检测太多？

1. 调整配置文件中的检测阈值
2. 减少 `--markers` 参数值
3. 确保背景简洁

### 程序运行慢？

1. 降低输入分辨率
2. 减少跟踪的标记数量
3. 关闭 `--show` 窗口（仅保存结果）

### 深度估计不准确？

1. **精确测量标记直径**：使用卡尺测量，确保 `--marker-diameter` 参数准确
2. **相机标定**：通过棋盘格标定获取准确焦距，使用 `--focal-length` 而非 `--fov`
3. **避免畸变区域**：使用 `--crop-border` 裁剪图像边缘
4. **保持垂直视角**：标记平面与相机光轴倾斜会导致误差

### 3D坐标不合理？

1. **检查相机位置**：确保 `--camera-position` 参数测量准确（单位：米）
2. **检查旋转角度**：确认俯仰角 `pitch` 符号正确（向下为正）
3. **确认坐标系定义**：世界坐标系原点是否正确设置
4. **验证地面高度**：检查Z=0是否确实为地平面

### 轨迹ID频繁丢失？

1. 增大 `--max-age`（如120-300帧）
2. 增大 `--memory-frames`（如1000-3000帧）
3. 放宽 `--match-threshold`（如300-500像素）
4. 使用预处理选项改善检测稳定性（`--temporal-smooth`, `--use-clahe`）

## 技术支持

- 项目路径：`d:\Learning Material\Git\WhyConID-py`
- 核心模块：`core/`
- 测试图片：`core/TEST/img_test.jpg`
