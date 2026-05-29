我来为你编写一个中英双语的 README 文档。

```markdown
# AI Robust Regression Tool / AI 鲁棒回归工具

[English](#english) | [中文](#chinese)

---

## English <a name="english"></a>

### Overview

An interactive GUI tool for robust regression analysis with three powerful algorithms:
- **Classic RANSAC** (C++ backend) - Robust against outliers
- **Neural Network** (PyTorch) - Deep learning approach
- **Symbolic Regression** (LASSO/Ridge) - Interpretable mathematical expressions

### Features

- **Interactive Plotting**: Click to add points, right-click to delete, middle-click to toggle exclude
- **Real-time Fitting**: Instant feedback when adding/modifying points
- **Outlier Detection**: Automatic identification of anomalies
- **CSV Import/Export**: Easy data exchange
- **3D Visualization**: Support for 3D plane fitting
- **Adjustable Grid**: Customize grid density, style, and transparency

### System Requirements

- **Python 3.8+** (with pip)
- **Visual Studio 2026** (for C++ compilation, if using RANSAC mode)

### Installation

1. **Clone or download** this repository

2. **Install Python dependencies** (using Aliyun mirror for faster download):
```bash
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

Or install manually:
```bash
pip install numpy torch scikit-learn PyQt6 matplotlib -i https://mirrors.aliyun.com/pypi/simple/
```

3. **Compile the C++ program** (required for Classic RANSAC mode):

   - Open `robust_regressor.cpp` in **Visual Studio 2026**
   - Create a new **Empty Project** (Console Application)
   - Add the `robust_regressor.cpp` file to the project
   - Build as **Release x64**
   - Copy the generated `RobustRegressor.exe` to the same directory as `regressor_ui.py`

   > **Note**: If you don't need RANSAC mode, you can skip this step and use Neural Network or Symbolic Regression modes instead.

### Project Structure

```
project/
├── regressor_ui.py          # Main GUI application (required)
├── RobustRegressor.exe      # C++ RANSAC backend (required for Classic mode)
├── robust_regressor.cpp     # C++ source code
├── train.py                 # Training/testing script
└── requirements.txt         # Python dependencies
```

### Usage

1. **Place files correctly**: Make sure `RobustRegressor.exe` and `regressor_ui.py` are in the **same directory**

2. **Run the application**:
```bash
python regressor_ui.py
```

3. **Basic Operations**:
   - **Left click on empty area**: Add a point
   - **Left click on a point**: Select it (turns blue)
   - **Right click on a point**: Delete it
   - **Middle click or Ctrl+Left click**: Toggle exclude status (turns orange)
   - **Scroll wheel**: Zoom in/out

4. **Select Regression Mode**:
   - **Classic RANSAC**: Fast C++ implementation, requires `RobustRegressor.exe`
   - **Neural Network**: Deep learning, slower but more flexible
   - **Symbolic (LASSO)**: Produces interpretable equations

5. **Adjust Settings**:
   - **Poly Degree**: Polynomial degree (0=auto)
   - **RANSAC Threshold**: Outlier detection threshold (-1=auto)
   - **Max Iter**: Maximum RANSAC iterations
   - **Grid Settings**: Adjust grid density, style, and transparency

### Development Environment

- **C++ Compiler**: Visual Studio 18 2026 (MSVC)
- **Project Type**: Empty Project → Console Application
- **Build Configuration**: Release x64

### Troubleshooting

**Error: "RobustRegressor.exe not found"**
- Solution: Compile the C++ code and place the exe in the same directory as the Python script
- Or switch to Neural Network / Symbolic Regression mode

**Mouse clicks not working**
- Make sure the toolbar's PAN/ZOOM mode is not active (click the "Home" button to reset)

**Slow performance when adding points**
- Adjust grid density to a lower value (e.g., 5-8)

### File Format

**CSV Input** (2D):
```csv
x,y
1.0,2.5
2.0,3.8
3.0,5.1
```

**CSV Input** (3D):
```csv
x,y,z
1.0,2.0,3.5
2.0,3.0,5.2
3.0,4.0,6.8
```

### License

This project is for research and educational purposes.

---

## 中文 <a name="chinese"></a>

### 概述

一个交互式的鲁棒回归分析 GUI 工具，提供三种强大的算法：
- **经典 RANSAC**（C++ 后端）- 对异常值鲁棒
- **神经网络**（PyTorch）- 深度学习方法
- **符号回归**（LASSO/Ridge）- 可解释的数学表达式

### 功能特点

- **交互式绘图**：点击添加点、右键删除、中键切换排除状态
- **实时拟合**：添加/修改点时即时反馈
- **异常点检测**：自动识别异常数据
- **CSV 导入/导出**：方便的数据交换
- **3D 可视化**：支持三维平面拟合
- **可调网格**：自定义网格密度、样式和透明度

### 系统要求

- **Python 3.8+**（含 pip）
- **Visual Studio 2026**（如需使用 RANSAC 模式）

### 安装步骤

1. **克隆或下载**本仓库

2. **安装 Python 依赖**（使用阿里云镜像加速）：
```bash
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

或手动安装：
```bash
pip install numpy torch scikit-learn PyQt6 matplotlib -i https://mirrors.aliyun.com/pypi/simple/
```

3. **编译 C++ 程序**（经典 RANSAC 模式需要）：

   - 在 **Visual Studio 2026** 中打开 `robust_regressor.cpp`
   - 创建一个新的**空项目**（控制台应用程序）
   - 将 `robust_regressor.cpp` 文件添加到项目中
   - 编译为 **Release x64**
   - 将生成的 `RobustRegressor.exe` 复制到与 `regressor_ui.py` 相同的目录

   > **注意**：如果不需要使用 RANSAC 模式，可以跳过此步骤，改用神经网络或符号回归模式。

### 项目结构

```
project/
├── regressor_ui.py          # 主 GUI 程序（必需）
├── RobustRegressor.exe      # C++ RANSAC 后端（经典模式需要）
├── robust_regressor.cpp     # C++ 源代码
├── train.py                 # 训练/测试脚本
└── requirements.txt         # Python 依赖列表
```

### 使用方法

1. **正确放置文件**：确保 `RobustRegressor.exe` 和 `regressor_ui.py` 在**同一目录**下

2. **运行程序**：
```bash
python regressor_ui.py
```

3. **基本操作**：
   - **左键点击空白处**：添加点
   - **左键点击数据点**：选中该点（变为蓝色）
   - **右键点击数据点**：删除该点
   - **中键或 Ctrl+左键**：切换排除状态（变为橙色）
   - **滚轮**：缩放视图

4. **选择回归模式**：
   - **经典 RANSAC**：快速 C++ 实现，需要 `RobustRegressor.exe`
   - **神经网络**：深度学习方法，较慢但更灵活
   - **符号回归**：生成可解释的数学方程

5. **调整设置**：
   - **多项式阶数**：多项式次数（0=自动）
   - **RANSAC 阈值**：异常点检测阈值（-1=自动）
   - **最大迭代**：RANSAC 最大迭代次数
   - **网格设置**：调整网格密度、样式和透明度

### 开发环境

- **C++ 编译器**：Visual Studio 18 2026（MSVC）
- **项目类型**：空项目 → 控制台应用程序
- **编译配置**：Release x64

### 常见问题

**错误："RobustRegressor.exe not found"**
- 解决方法：编译 C++ 代码并将 exe 放在 Python 脚本同目录下
- 或者切换到神经网络/符号回归模式

**鼠标点击无效**
- 确保工具栏的平移/缩放模式未激活（点击"Home"按钮重置）

**添加点时卡顿**
- 将网格密度调低（例如 5-8）

### 文件格式

**CSV 输入**（2D）：
```csv
x,y
1.0,2.5
2.0,3.8
3.0,5.1
```

**CSV 输入**（3D）：
```csv
x,y,z
1.0,2.0,3.5
2.0,3.0,5.2
3.0,4.0,6.8
```

### 许可证

本项目仅用于研究和教育目的。

---

## Contact / 联系方式

For questions or issues, please open an issue on GitHub.
如有问题，请在 GitHub 上提交 Issue。
```

同时创建一个 `requirements.txt` 文件：

```txt
# Python dependencies for AI Robust Regression Tool
# Install using: pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

numpy>=1.21.0
torch>=2.0.0
scikit-learn>=1.0.0
PyQt6>=6.4.0
matplotlib>=3.5.0
```

以及一个快速启动脚本 `run.bat`（Windows 用户）：

```batch
@echo off
chcp 65001 >nul
title AI Robust Regression Tool

echo ========================================
echo    AI Robust Regression Tool
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo Please install Python 3.8+
    pause
    exit /b 1
)

echo [INFO] Starting application...
echo [INFO] Make sure RobustRegressor.exe is in the same directory
echo.

python regressor_ui.py

pause
```

## 使用说明总结

用户只需要：

1. 确保 `RobustRegressor.exe` 和 `regressor_ui.py` 在**同一文件夹**
2. 安装 Python 依赖（使用阿里云镜像）
3. 运行 `python regressor_ui.py` 或双击 `run.bat`

**不需要打包成单个 exe**，因为：
- 打包后文件巨大（200-400 MB）
- 首次启动慢
- 开发调试不便

这样用户可以直接运行 Python 脚本，同时 C++ 程序作为外部工具被调用。
