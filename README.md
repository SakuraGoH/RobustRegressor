# RobustRegressor
# AI 鲁棒回归工具 / AI Robust Regression Tool

<div align="center">
  
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyQt6](https://img.shields.io/badge/PyQt6-6.4+-green.svg)](https://www.riverbankcomputing.com/software/pyqt/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

## 📖 简介 / Introduction

**中文**：AI 鲁棒回归工具是一个强大的交互式数据拟合软件，集成了三种先进的回归算法，支持 2D/3D 数据的异常点检测和可视化拟合。用户可以通过鼠标交互添加、删除和标记数据点，实时查看拟合结果。

**English**：AI Robust Regression Tool is a powerful interactive data fitting software that integrates three advanced regression algorithms, supporting outlier detection and visual fitting for 2D/3D data. Users can interactively add, delete, and mark data points with real-time fitting visualization.

---

## ✨ 主要特性 / Features

| 特性 / Feature | 说明 / Description |
|---------------|-------------------|
| 🎯 **三种回归模式** | 经典 RANSAC / 神经网络 / 符号回归 |
| 📊 **2D/3D 支持** | 平面拟合和曲面拟合 |
| 🖱️ **交互式操作** | 鼠标添加、删除、标记异常点 |
| 📈 **实时拟合** | 数据变化时自动重新拟合 |
| 🎨 **网格调节** | 可调节网格密度、样式和透明度 |
| 📁 **数据导入导出** | 支持 CSV 文件导入/导出 |
| 🔍 **异常检测** | 自动识别并高亮异常点 |

---

## 🚀 快速开始 / Quick Start

### 中文

#### 1. 环境要求
- Python 3.8 或更高版本
- Windows 10/11 (推荐) 或 Linux/macOS

#### 2. 安装依赖

使用阿里云镜像快速安装：

```bash
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

或者手动安装：

```bash
pip install numpy torch scikit-learn PyQt6 matplotlib -i https://mirrors.aliyun.com/pypi/simple/
```

#### 3. 编译 C++ RANSAC 程序（可选）

如需使用经典 RANSAC 模式，需要编译 C++ 程序：

**Visual Studio 方法**：
1. 打开 Visual Studio，创建新的 C++ 控制台应用程序
2. 将 `robust_regressor.cpp` 代码复制到主文件
3. 编译生成 Release 版本
4. 将 `RobustRegressor.exe` 放在与 `regressor_ui.py` 同一目录

**MinGW 方法**：
```bash
g++ -O2 -std=c++17 robust_regressor.cpp -o RobustRegressor.exe
```

#### 4. 运行程序

```bash
python regressor_ui.py
```

### English

#### 1. Requirements
- Python 3.8 or higher
- Windows 10/11 (recommended) or Linux/macOS

#### 2. Install Dependencies

Using Aliyun mirror for fast installation:

```bash
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

Or manually:

```bash
pip install numpy torch scikit-learn PyQt6 matplotlib
```

#### 3. Compile C++ RANSAC Program (Optional)

To use Classic RANSAC mode, compile the C++ program:

**Visual Studio method**:
1. Open Visual Studio, create a new C++ console application
2. Copy `robust_regressor.cpp` code to the main file
3. Build Release version
4. Place `RobustRegressor.exe` in the same directory as `regressor_ui.py`

**MinGW method**:
```bash
g++ -O2 -std=c++17 robust_regressor.cpp -o RobustRegressor.exe
```

#### 4. Run the Program

```bash
python regressor_ui.py
```

---

## 🖱️ 鼠标操作 / Mouse Controls

| 操作 / Action | 功能 / Function |
|--------------|-----------------|
| **左键点击空白处** | 添加数据点 |
| **左键点击数据点** | 选中数据点 |
| **右键点击数据点** | 删除数据点 |
| **中键 / Ctrl+左键** | 切换排除/包含状态 |
| **滚轮** | 缩放视图 |
| **工具栏按钮** | 平移、缩放、重置视图 |

---

## 📁 文件结构 / File Structure

```
project/
├── regressor_ui.py          # 主程序 GUI
├── robust_regressor.cpp     # C++ RANSAC 实现
├── train.py                 # 训练和测试脚本
├── requirements.txt         # Python 依赖列表
├── RobustRegressor.exe      # 编译后的 C++ 程序（可选）
└── README.md                # 说明文档
```

---

## 🧠 算法说明 / Algorithm Description

### 1. 经典 RANSAC / Classic RANSAC

**中文**：基于随机抽样一致性算法，通过迭代采样和模型验证来识别异常点。适合处理大量异常值的数据集。

**English**：Based on Random Sample Consensus algorithm, identifies outliers through iterative sampling and model verification. Suitable for datasets with a large number of outliers.

### 2. 神经网络 / Neural Network

**中文**：使用多层感知机（MLP）进行非线性回归，自动学习数据中的复杂模式。网络结构：128 → 64 → 32 → 1。

**English**：Uses Multi-Layer Perceptron (MLP) for nonlinear regression, automatically learning complex patterns in data. Network structure: 128 → 64 → 32 → 1.

### 3. 符号回归 / Symbolic Regression (LASSO)

**中文**：基于 LASSO/Ridge 回归，自动构建包含多项式、三角函数、指数函数等基函数的稀疏模型，输出可解释的数学表达式。

**English**：Based on LASSO/Ridge regression, automatically builds sparse models with basis functions including polynomials, trigonometric functions, exponential functions, etc., outputting interpretable mathematical expressions.

---

## 🎮 使用指南 / User Guide

### 中文

#### 基本操作流程

1. **添加数据点**：
   - 方式一：在左侧输入框输入坐标，点击 "Add Point"
   - 方式二：在绘图区左键点击空白处

2. **选择回归模式**：
   - Classic RANSAC：需要 C++ 程序支持
   - Neural Network：神经网络模式
   - Symbolic (LASSO)：符号回归模式

3. **调节参数**：
   - Poly Degree：多项式阶数（仅 RANSAC）
   - RANSAC Threshold：异常检测阈值
   - Max Iter：最大迭代次数

4. **网格设置**：
   - Grid Density：调节网格疏密（2-20）
   - Grid Style：选择网格样式
   - Grid Alpha：调节网格透明度

5. **导出结果**：
   - 点击 "Export Result" 保存拟合方程和统计数据

#### 数据格式

CSV 文件格式示例：

**2D 数据**：
```csv
x,y
1.0,2.5
2.0,3.8
3.0,5.2
```

**3D 数据**：
```csv
x,y,z
1.0,2.0,5.5
2.0,3.0,7.2
3.0,4.0,8.9
```

### English

#### Basic Workflow

1. **Add Data Points**:
   - Method 1: Enter coordinates in left panel, click "Add Point"
   - Method 2: Left-click on empty area in the plot

2. **Select Regression Mode**:
   - Classic RANSAC: Requires C++ program
   - Neural Network: Neural network mode
   - Symbolic (LASSO): Symbolic regression mode

3. **Adjust Parameters**:
   - Poly Degree: Polynomial degree (RANSAC only)
   - RANSAC Threshold: Outlier detection threshold
   - Max Iter: Maximum iterations

4. **Grid Settings**:
   - Grid Density: Adjust grid density (2-20)
   - Grid Style: Select grid style
   - Grid Alpha: Adjust grid transparency

5. **Export Results**:
   - Click "Export Result" to save fitting equation and statistics

#### Data Format

CSV file format example:

**2D Data**:
```csv
x,y
1.0,2.5
2.0,3.8
3.0,5.2
```

**3D Data**:
```csv
x,y,z
1.0,2.0,5.5
2.0,3.0,7.2
3.0,4.0,8.9
```

---

## 🔧 打包成独立 EXE / Packaging as Standalone EXE

### 中文

使用 PyInstaller 打包成单个可执行文件：

```bash
# 安装 PyInstaller
pip install pyinstaller

# 打包（包含 C++ 程序）
pyinstaller --onefile --windowed --name=AI_Robust_Regression --add-data "RobustRegressor.exe;." regressor_ui.py
```

打包后的文件位于 `dist/AI_Robust_Regression.exe`，可直接分发使用。

### English

Use PyInstaller to package as a single executable file:

```bash
# Install PyInstaller
pip install pyinstaller

# Package (including C++ program)
pyinstaller --onefile --windowed --name=AI_Robust_Regression --add-data "RobustRegressor.exe;." regressor_ui.py
```

The packaged file is located at `dist/AI_Robust_Regression.exe` and can be distributed directly.

---

## 🎨 颜色说明 / Color Legend

| 颜色 / Color | 含义 / Meaning |
|-------------|---------------|
| 🟢 **绿色** | 正常点 / Normal point (Inlier) |
| 🔴 **红色** | AI 检测异常点 / AI-detected outlier |
| 🟠 **橙色** | 手动排除点 / Manually excluded point |
| 🔵 **蓝色** | 选中的点 / Selected point |
| 🔴 **红色曲线** | 拟合曲线/曲面 / Fitted curve/surface |

---

## 📋 快捷键 / Shortcuts

| 快捷键 / Shortcut | 功能 / Function |
|------------------|-----------------|
| `Ctrl + +` | 增加网格密度 / Increase grid density |
| `Ctrl + -` | 减少网格密度 / Decrease grid density |

---

## ⚠️ 常见问题 / FAQ

### 中文

**Q1: 提示 "RobustRegressor.exe not found"**

A: 经典 RANSAC 模式需要 C++ 程序支持。请：
- 切换到神经网络或符号回归模式
- 或者编译 C++ 程序并放在正确位置

**Q2: 打包后的 EXE 文件很大（200-400 MB）**

A: 这是正常的，因为包含了 PyTorch 等大型依赖库。如果不需要神经网络功能，可以移除 torch 依赖。

**Q3: 鼠标点击无法添加点**

A: 请检查工具栏是否处于平移/缩放模式（按钮高亮）。点击工具栏的 "Pan" 或 "Zoom" 按钮退出该模式。

**Q4: 网格调节导致卡顿**

A: 已优化使用 `MaxNLocator` 自动刻度，如果仍然卡顿，请降低网格密度或选择 "Major" 模式。

### English

**Q1: Error "RobustRegressor.exe not found"**

A: Classic RANSAC mode requires the C++ program. Please:
- Switch to Neural Network or Symbolic regression mode
- Or compile the C++ program and place it in the correct location

**Q2: Packaged EXE file is large (200-400 MB)**

A: This is normal as it includes large dependencies like PyTorch. If you don't need neural network functionality, you can remove torch dependency.

**Q3: Cannot add points with mouse click**

A: Check if the toolbar is in Pan/Zoom mode (button highlighted). Click the "Pan" or "Zoom" button to exit that mode.

**Q4: Lag when adjusting grid**

A: Optimized with `MaxNLocator` for automatic ticks. If still lagging, reduce grid density or select "Major" mode.

---

## 📄 许可证 / License

MIT License

---

## 📞 联系方式 / Contact

如有问题或建议，欢迎提 Issue 或 Pull Request。

For issues or suggestions, please submit an Issue or Pull Request.

---

<div align="center">
  
**享受使用！/ Enjoy!** 🎉

</div>

---

## 📦 requirements.txt

创建 `requirements.txt` 文件：

```txt
numpy>=1.21.0
torch>=2.0.0
scikit-learn>=1.0.0
PyQt6>=6.4.0
matplotlib>=3.5.0
```
