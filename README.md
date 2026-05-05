# 数字图像处理 —— 广东省车牌识别系统

基于 OpenCV 的传统图像处理方法实现的中国车牌自动识别系统，专注于**广东省车牌**（粤字头）的检测与识别。

## 功能特性

- **车牌定位**：基于 Sobel/Laplacian 边缘检测 + HSV 颜色空间形态学处理的混合策略
- **透视矫正**：针对倾斜车牌的 4 点透视变换（Homography），自动校正斜拍角度
- **字符分割**：上下分区 + 自适应阈值 + 抗粘连裁剪的精细化分割算法
- **字符识别**：模板匹配（TM_CCOEFF_NORMED）+ 拓扑学校正（区分 6/8/B/G）
- **多车牌检测**：支持单张图片中多辆车的车牌同时检测（NMS 去重）
- **遮挡检测**：智能识别局部遮挡/污渍，并给出警告提示
- **多场景支持**：正面、倾斜、正常角度、遮挡/干扰等多种场景

## 项目结构

```
├── cover_new.py          # 正面车牌识别（含遮挡检测 + 抗粘连）
├── slant_new.py          # 倾斜车牌识别（透视矫正 + 混合分割）
├── normal.py             # 正常角度车牌识别
├── multiple.py           # 多车牌同时检测
├── optimize.py           # 遮挡/干扰场景优化版
├── template_new.py       # 中文字符模板生成器（粗体）
│
├── images/               # 测试图片集
│   ├── cover/            #   正面车牌
│   ├── slant/            #   倾斜车牌
│   ├── normal/           #   正常角度车牌
│   └── distraction/      #   遮挡/干扰车牌
│
├── templates_cn/         # 中文字符模板（标准粗细）
└── templates_bold/       # 中文字符模板（粗体）
```

## 环境依赖

- Python 3.x
- OpenCV (`opencv-python`)
- NumPy
- Pillow（仅模板生成需要）

安装依赖：

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 生成中文模板（首次运行需要）

```bash
python template_new.py          # 生成粗体中文字符模板
```

> 注意：模板生成需要系统中安装 **SimHei（黑体）** 字体，默认路径 `C:/Windows/Fonts/simhei.ttf`。

### 2. 运行识别

```bash
# 正面车牌
python cover_new.py

# 倾斜车牌
python slant_new.py

# 正常角度车牌
python normal.py

# 多车牌检测
python multiple.py

# 遮挡干扰场景
python optimize.py
```

每个脚本底部 `if __name__ == "__main__":` 中可修改 `img_path` 来指定目标图片。

程序运行后会依次弹出多个 OpenCV 调试窗口展示各处理阶段，按任意键关闭。

## 技术流程

```
输入图片
  │
  ▼
预处理（灰度化 → 边缘检测 → HSV白掩码 → 形态学闭操作）
  │
  ▼
车牌定位（轮廓筛选 → 纹理密度验证 → 白像素比例验证 → NMS去重）
  │
  ▼
透视矫正（4点排序 → Homography变换）  ← 仅 slant_new/cover_new
  │
  ▼
字符分割（上下分区 → 自适应阈值 → 抗粘连裁剪 → C位排序 → IOU去重）
  │
  ▼
字符识别（模板匹配 → 置信度阈值 → 拓扑校正 → 城市名补全）
  │
  ▼
输出结果
```

### 字符识别补充

- **中文字符**：模板匹配广州、东莞、佛山等广东省城市简称
- **英数字符**：0-9, A-H, J-N, P, Q-Z（排除不易混淆的 I/O）
- **拓扑校正**：通过计算二值图像中的"洞数"（欧拉数）区分 6/8/B/G

## 适用场景

| 脚本 | 适用场景 |
|------|---------|
| `cover_new.py` | 正面拍摄、车牌无倾斜 |
| `slant_new.py` | 车牌有明显倾斜/斜角 |
| `normal.py` | 标准角度、无需透视矫正 |
| `multiple.py` | 图片中有多辆车 |
| `optimize.py` | 车牌有遮挡、污渍、干扰物 |
