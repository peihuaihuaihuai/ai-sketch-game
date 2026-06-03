# 🎨 AI 速写猜谜 (AI Sketch Guesser)

基于 CNN 的手绘草图识别 Web 应用。用户在画布上作画，AI 实时识别所画物体类别。

**支持类别**: ✈️ 飞机 (airplane) · 🚗 汽车 (car) · 🐱 猫 (cat) · 🐶 狗 (dog) · 🏠 房子 (house) · 🌲 树 (tree)

---

## 项目结构

```
ai-sketch-game/
├── app.py                      # Flask Web 应用入口
├── requirements.txt            # Python 依赖
├── api/
│   ├── __init__.py
│   └── predict.py              # 推理 API 逻辑（模型加载、预测路由）
├── model/
│   ├── __init__.py
│   ├── model.py                # CNN 模型定义（QuickDrawResNet）
│   ├── preprocessing.py        # 图像预处理（归一化、张量转换）
│   ├── dataset.py              # QuickDraw 数据集加载器
│   ├── stroke_graph.py         # 笔画图构建 & GNN/Hybrid 模型
│   ├── train.py                # CNN 训练脚本
│   ├── train_gnn.py            # GNN 训练脚本
│   ├── diagnostics.py          # 模型诊断脚本
│   ├── quickdraw_cnn.pth       # 训练好的 CNN 模型权重
│   └── training_log.txt        # 训练日志
├── data/
│   └── raw/                    # QuickDraw 原始 .npy 数据
├── static/
│   ├── index.html              # 静态前端（英文版）
│   ├── css/
│   │   └── style.css           # Claude Design Style 样式
│   └── js/
│       ├── canvas.js           # 画布绘图 & 预处理管线
│       ├── api.js              # 后端 API 通信
│       ├── visualization.js    # 预测结果可视化
│       └── history.js          # 绘画历史记录
├── templates/
│   └── index.html              # Flask 模板（中文版）
└── tests/
    ├── test_model.py           # 模型单元测试
    ├── test_api.py             # API 单元测试
    ├── test_stroke_graph.py    # 笔画图构建测试
    └── test_class_separation.py# 类别分离度测试
```

---

## 快速开始

### 环境要求

- Python 3.10+
- 操作系统: Windows / macOS / Linux

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动服务

```bash
python app.py
```

服务启动后访问: **http://localhost:5000**

### 3. 使用

1. 在画布上绘制 6 种类别之一的草图
2. 停止绘画后自动触发预测（可调节延迟），或点击 **🔮 预测** 按钮
3. 右侧面板显示 Top-5 预测结果和置信度趋势

---

## 技术架构

### 模型

| 模型 | 架构 | 参数量 | 验证精度 |
|------|------|--------|---------|
| **CNN** (当前使用) | QuickDrawResNet | 308,911 | **94.33%** |
| GNN | StrokeGNN (纯 PyTorch 消息传递) | ~59K | 未训练 |
| Hybrid | CNN + GNN 联合推理 | ~408K | 未微调 |

CNN 模型基于 ResNet 残差结构，输入为 28×28 灰度位图（白笔画/黑背景），输出 6 类 logits。

### 前端预处理管线

```
用户画布 (400×400, 8px 笔宽, 白底黑字)
  → 质心 (center-of-mass) 定位
  → 方形裁剪 + 15% 边距
  → 双线性缩放至 28×28
  → 自适应阈值二值化（目标密度 25%）
  → 784 浮点数组 [0, 1]（白笔画/黑背景）
```

### API 接口

**`POST /predict`**

请求体:
```json
{
    "pixels": [0.0, 0.12, ..., 0.98],
    "strokes": [[{"x": 0.2, "y": 0.3}, ...], ...]
}
```

响应:
```json
{
    "top5": [
        {"label": "airplane", "probability": 0.9234},
        {"label": "car",     "probability": 0.0512},
        ...
    ],
    "latency_ms": 2.145,
    "model": "resnet"
}
```

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 主页面 |
| `/predict` | POST | 草图预测 |
| `/reset` | POST | 重置服务端状态 |
| `/health` | GET | 健康检查 |

---

## 运行测试

```bash
# 运行全部测试（92 个用例）
python -m pytest tests/ -v

# 运行单个模块
python -m pytest tests/test_model.py -v
python -m pytest tests/test_api.py -v
python -m pytest tests/test_stroke_graph.py -v
python -m pytest tests/test_class_separation.py -v
```

---

## 模型训练

### CNN

```bash
python model/train.py
```

- 数据集: Google QuickDraw 官方 .npy 文件（120,000 样本）
- 训练/验证: 80/20 划分
- 优化器: Adam, 初始学习率 0.001, cosine 退火
- 早停: 5 epoch 无提升则停止
- 最佳结果: epoch 10, val_acc = 94.33%

### GNN（未训练）

```bash
python model/train_gnn.py
```

### 模型诊断

```bash
python model/diagnostics.py
```

---

## 依赖

| 包 | 版本 | 用途 |
|----|------|------|
| Flask | 3.0 | Web 框架 |
| PyTorch | 2.1 | 深度学习推理 |
| NumPy | 1.24 | 数值计算 |
| Pillow | 10.0 | 图像处理 |
| Pytest | 7.4 | 单元测试 |


## 课程信息

课程名称：神经网络与深度学习

项目名称：AI Sketch Game

开发语言：Python

框架：Flask + PyTorch