# WhyConID-py 核心

WhyConID 圆形标记检测与识别系统的 Python 实现。

## 📁 模块结构

```
core/
├── detectors/          # 标记检测模块
│   ├── base_detector.py      # 抽象基类
│   ├── circle_detect.py      # 主要圆形检测器
│   └── offcircle_detect.py   # 椭圆/离轴检测器
├── id_generation/      # ID 编码/解码
│   └── necklace.py           # Necklace 风格的 ID 生成
├── processing/         # 图像处理
│   ├── image_processor.py    # 预处理工具
│   └── lgmd.py              # 运动检测
├── geometry/           # 几何算法
│   ├── circle_fitting.py     # 圆/椭圆拟合
│   ├── transformation.py     # 坐标变换
│   └── calibration.py       # 相机标定
├── utils/              # 工具
│   ├── config.py            # 配置管理
│   └── logger.py            # 日志工具
└── main.py            # 命令行入口
```

## 🚀 快速开始

### 安装

```bash
cd core
pip install -r requirements.txt
```

### 基本用法

```python
from detectors.circle_detect import CircleDetector
from id_generation.necklace import CNecklace
import cv2

# 读取图像
image = cv2.imread('test.jpg')

# 初始化检测器
detector = CircleDetector(
  width=image.shape[1],
  height=image.shape[0],
  num_bots=1
)

# 检测标记
segments = detector.detect(image)

# 打印结果
for seg in segments:
  print(f"Marker at ({seg.x:.1f}, {seg.y:.1f}), ID: {seg.ID}")
```

### 命令行

```bash
# 处理图像
python main.py input.jpg --show

# 处理视频
python main.py input.mp4 --output output.mp4 --markers 3

# 使用摄像头
python main.py 0 --show --debug 1
```

## 🔧 配置

创建一个 `config.json` 文件：

```json
{
  "detection": {
  "threshold": 128,
  "min_size": 20,
  "circular_tolerance": 0.3
  },
  "camera": {
  "width": 640,
  "height": 480,
  "fps": 30
  },
  "marker": {
  "diameter": 0.05,
  "necklace_bits": 5,
  "num_markers": 1
  }
}
```

## 📚 API 参考

### CircleDetector

用于圆形标记检测与 ID 识别的主检测器类。

方法：

- `detect(image)`：在图像中检测标记，返回 Segment 对象列表
- `reset()`：重置检测器状态

### CNecklace

用于旋转不变识别的 Necklace ID 编码/解码器。

方法：

- `get_id(code)`：获取位模式的 ID 信息
- `decode_sequence(sequence)`：将二进制序列解码为 ID
- `extract_from_points(points, center)`：从圆点提取 ID

### 坐标变换

函数：

- `fit_circle_algebraic(points)`：快速代数圆拟合
- `fit_circle_nonlinear(points)`：精确几何拟合
- `fit_ellipse(points)`：椭圆拟合

## 📖 算法细节

参见 `../docs/instruction/` 中的文档：

- `ALGORITHM_DOCUMENTATION.md`：算法理论与数学
- `PY_REPRODUCTION.md`：Python 重现指南
- `PROJECT_STRUCTURE.md`：原始 C# 项目结构

## 🔗 参考文献

基于 Qinbing Fu（2017 年 1 月）的 WhyConID，实现参考：

- [1] Krajnik, Nitsche 等：A practical multirobot localization system. Journal of Intelligent and Robotic Systems, 2014.
- [2] Peter Lightbody 等：A Versatile High-Performance Visual Fiducial Marker Detection System with Scalable Identity Encoding. SAC 2017.
