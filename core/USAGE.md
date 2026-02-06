# WhyConID-py 使用文档

WhyConID-py 是一个基于Python的圆形标记检测和识别系统，使用内外环算法检测 WhyConID 标记。

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

#### 轨迹追踪选项 (新功能)

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

**轨迹追踪示例：**

```bash
# 基础轨迹追踪
python main.py video.mp4 --track --show --markers 2

# 保存轨迹数据
python main.py video.mp4 --track --save-trajectory --markers 3

# 持久化彩色轨迹 + 视频导出
python main.py video.mp4 --track --persistent-trajectory --color-trajectory --output tracked.mp4

# 完整功能演示
python main.py video.mp4 --track --save-trajectory --persistent-trajectory --color-trajectory --output result.mp4 --markers 5
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

#### 标注图片/视频

图片/视频上的可视化标记：

- **绿色十字**：标记中心点
- **蓝色矩形框**：边界框
- **黄色文字**：ID（Track ID或Marker ID）、坐标、圆度、面积
- **橙色轨迹线**（启用追踪时）：标记运动轨迹
- **彩色轨迹**（启用 `--color-trajectory`）：每条轨迹不同颜色
- **白色面板**：检测统计和时间戳

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

## 技术支持

- 项目路径：`d:\Learning Material\Git\WhyConID-py`
- 核心模块：`core/`
- 测试图片：`core/TEST/img_test.jpg`
