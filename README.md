# WhyConID-py

Python 复现与扩展版的 WhyConID——圆形/环形标记检测、ID 生成与轨迹跟踪工具套件。

本仓库实现了原始 C# WhyConID 的核心检测、ID 解码、跟踪与可视化，并新增命令行工具与可选的 HTTP API 服务（用于前端集成）。

**重要路径**

- `core/`：核心实现（检测、ID、生成功能、跟踪、可视化）。命令行入口：`core/main.py`。
- `api/`：可选 REST API 服务（FastAPI），将 `core` 功能暴露给前端或外部系统。
- `web/`：前端项目（Vue + Vite），可选构建并由 `api` 挂载 [开发中]。

主要特性

- 圆/椭圆检测（motion-aware 自适应阈值和轮廓筛选）
- `Necklace` 风格 ID 生成与解码
- 多目标卡尔曼跟踪（ID 恢复、持久轨迹、彩色轨迹）
- 轨迹预测（单步箭头、多步虚线、预测误差可视化）
- 命令行处理（图像/视频/摄像头）与可视化输出
- 可选 API 服务：图像/视频上传、结果返回、标注视频与轨迹下载

快速开始

1) 克隆并进入仓库

```bash
git clone <repo-url>
cd "d:/Learning Material/Git/WhyConID-py"
```

2) 创建虚拟环境并安装 core 依赖

```bash
cd core
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

3) 使用命令行处理（示例：检测视频并显示实时窗口）

```bash
# 处理视频并显示（带轨迹和预测）
python core/main.py TEST/test_double_colias.mp4 --track --show --show-prediction --prediction-steps 5 --persistent-trajectory --color-trajectory --output result.mp4
```

4) 启动 API 服务（可选）

```bash
cd api
.venv\Scripts\activate       # 使用同一虚拟环境或新建一个
pip install -r requirements.txt
python server.py --mode api    # 或 --mode full / --serve-frontend
# 打开 API 文档: http://localhost:8000/docs
```

前端（开发）

前端位于 `web/`，使用 Vite + Vue：

```bash
cd web
npm install
npm run dev       # 本地开发
npm run build     # 生产构建，构建结果放到 web/dist
```

将前端集成到 API（完整模式）

- 在 `api/server.py`启动时使用 `--mode full`或 `--serve-frontend`，服务会尝试挂载 `web/dist` 目录作为静态站点。

工程结构（摘要）

```
WhyConID-py/
├── api/                # REST API（FastAPI）
├── core/               # 核心模块（检测、跟踪、可视化）
├── web/                # 前端（Vue + Vite）
└── WhyConID/           # 原始 C# 项目（参考）
```

开发说明

- `core/` 保持为可复用模块，`api/` 与 `core/` 共享同一实现，便利不同调用方式（CLI / HTTP）。
- 在修改核心算法后，请优先在 `core` 下添加单元测试（`core/test_core.py`）并验证再集成到 API。

常见命令速查

```bash
# 运行单张图像检测并显示
python core/main.py test.jpg --show

# 批量处理视频并保存标注视频
python core/main.py input.mp4 --track --output annotated.mp4

# 启动 API 服务（开发）
cd api
python server.py --mode api --reload

# 启动 API + 前端（需先构建前端）
python server.py --mode full
```

贡献与许可

- 欢迎提 PR（功能增强、性能优化、前端改进）。
- 若需将项目开源，请在仓库根添加许可证（例如 MIT）。

---

![Stars](https://www.star-history.com/#TiiJeiJ8/WhyConID-py&type=date&legend=top-left)
