# AI Robust Regression Tool

## English Version

### Overview
An interactive GUI tool for robust regression with RANSAC, Neural Network, and Symbolic LASSO methods. Supports both 2D and 3D data fitting with outlier detection and manual point editing.

### Features
- **Three regression modes**: Classic RANSAC, Neural Network, Symbolic LASSO
- **2D & 3D visualization**: Interactive plots with zoom/pan controls
- **Interactive point editing**: Add, delete, select, and manually exclude points
- **Real-time fitting**: Automatic re-fitting when data changes
- **CSV import/export**: Load data from CSV, export results with equations
- **Adjustable grid**: Control grid density, style, and transparency

### Requirements
- Python 3.8 or higher
- Required packages: `numpy` `torch` `scikit-learn` `PyQt6` `matplotlib`

### Installation
```bash
pip install numpy torch scikit-learn PyQt6 matplotlib -i https://mirrors.aliyun.com/pypi/simple/
```

### Project Structure
```
project_folder/
├── regressor_ui.py          # Main GUI program (run this)
├── RobustRegressor.exe      # C++ RANSAC program (required for Classic mode)
├── robust_regressor.cpp     # C++ source code (optional, for recompilation)
└── train.py                 # Training script (optional)
```

### Usage
1. Place `regressor_ui.py` and `RobustRegressor.exe` in the **same directory**
2. Run: `python regressor_ui.py`
3. Select mode (2D/3D) and regression method
4. Add points by clicking on the plot or using input fields
5. Right-click to delete points, Middle-click/Ctrl+click to exclude points
6. Use toolbar buttons to zoom/pan

### Important Notes
- **Classic RANSAC mode** requires `RobustRegressor.exe` in the same folder as the Python script
- **Neural Network** and **Symbolic LASSO** modes work without the C++ executable
- The C++ program was compiled with **Visual Studio 18 2026** as an **empty project** (console application)

### C++ Compilation (if needed)
If you need to recompile `robust_regressor.cpp`:
1. Open Visual Studio 18 2026
2. Create a new **Empty C++ Console Application** project
3. Copy the entire content of `robust_regressor.cpp` into the main source file
4. Build as Release/x64
5. Copy `RobustRegressor.exe` to the Python script directory

### Mouse Operations
| Action | Effect |
|--------|--------|
| Left click (empty area) | Add point |
| Left click (on point) | Select point |
| Right click (on point) | Delete point |
| Middle click / Ctrl+Left click | Toggle exclude/include |
| Scroll | Zoom in/out |

---

## 中文版本

### 概述
一款交互式鲁棒回归拟合工具，支持 RANSAC、神经网络和符号 LASSO 三种回归方法，可处理 2D 和 3D 数据，具有异常点检测和手动点编辑功能。

### 功能特点
- **三种回归模式**：经典 RANSAC、神经网络、符号 LASSO
- **2D & 3D 可视化**：交互式绘图，支持缩放/平移
- **交互式点编辑**：添加、删除、选中、手动排除点
- **实时拟合**：数据变化时自动重新拟合
- **CSV 导入/导出**：从 CSV 加载数据，导出结果和方程
- **可调网格**：控制网格密度、样式和透明度

### 环境要求
- Python 3.8 或更高版本
- 必需包：`numpy` `torch` `scikit-learn` `PyQt6` `matplotlib`

### 安装方法
```bash
pip install numpy torch scikit-learn PyQt6 matplotlib -i https://mirrors.aliyun.com/pypi/simple/
```

### 项目结构
```
项目文件夹/
├── regressor_ui.py          # 主 GUI 程序（运行此文件）
├── RobustRegressor.exe      # C++ RANSAC 程序（经典模式需要）
├── robust_regressor.cpp     # C++ 源代码（可选，用于重新编译）
└── train.py                 # 训练脚本（可选）
```

### 使用方法
1. 将 `regressor_ui.py` 和 `RobustRegressor.exe` 放在**同一目录**下
2. 运行：`python regressor_ui.py`
3. 选择维度（2D/3D）和回归方法
4. 通过点击绘图区域或使用输入框添加点
5. 右键删除点，中键/Ctrl+左键切换排除/包含
6. 使用工具栏按钮进行缩放/平移

### 重要说明
- **经典 RANSAC 模式**需要 `RobustRegressor.exe` 与 Python 脚本在同一文件夹
- **神经网络**和**符号 LASSO**模式无需 C++ 可执行文件即可运行
- C++ 程序使用 **Visual Studio 18 2026** 以**空项目**（控制台应用程序）编译

### C++ 编译（如需重新编译）
如果需要重新编译 `robust_regressor.cpp`：
1. 打开 Visual Studio 18 2026
2. 新建**空 C++ 控制台应用程序**项目
3. 将 `robust_regressor.cpp` 的全部内容复制到主源文件中
4. 以 Release/x64 模式生成
5. 将生成的 `RobustRegressor.exe` 复制到 Python 脚本目录

### 鼠标操作说明
| 操作 | 效果 |
|------|------|
| 左键（空白处） | 添加点 |
| 左键（点上） | 选中点 |
| 右键（点上） | 删除点 |
| 中键 / Ctrl+左键 | 切换排除/包含 |
| 滚轮 | 缩放 |
