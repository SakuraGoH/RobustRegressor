import sys
import os
import subprocess
import csv
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LassoCV, Lasso, RidgeCV
from sklearn.preprocessing import StandardScaler
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QDoubleSpinBox, QSpinBox, QSlider,
    QFileDialog, QTableWidget, QTableWidgetItem, QGroupBox,
    QSplitter, QLineEdit, QCheckBox, QMessageBox, QHeaderView,
    QFrame, QInputDialog
)
from PyQt6.QtCore import Qt

import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt

# 字体配置
import matplotlib.font_manager as fm
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator, MaxNLocator


# ==================== UI 文本 ====================
UI_TEXTS = {
    "title": "AI Robust Regression (RANSAC / Neural / Symbolic)",
    "file_ops": "File Operations",
    "import_csv": "Import CSV",
    "export_result": "Export Result",
    "clear_all": "Clear All",
    "model_settings": "Model Settings",
    "mode": "Mode:",
    "dimension": "Dimension:",
    "poly_degree": "Poly Degree:",
    "ransac_thresh": "RANSAC Threshold:",
    "max_iter": "Max Iter:",
    "realtime": "Real-time fitting",
    "run_fit": "Run Fitting",
    "add_point": "Add Point",
    "x": "x",
    "y": "y",
    "z_3d": "z (3D mode)",
    "fit_info": "Fit Info",
    "equation": "Equation:",
    "na": "N/A",
    "r2_rmse": "R2: -- | RMSE: --",
    "counts": "Total: 0 | Inliers: 0 | Outliers: 0 | Excluded: 0",
    "outliers_list": "Outliers / Excluded",
    "index": "Index",
    "coords": "Coords",
    "reason": "Reason",
    "ready": "Ready",
    "classic_ransac": "Classic RANSAC",
    "neural_net": "Neural Network",
    "symbolic_lasso": "Symbolic (LASSO)",
    "2d": "2D",
    "3d": "3D",
    "auto_classic": "0 = auto (classic only)",
    "auto_thresh": "-1 = auto (classic only)",
    "input_error": "Input Error",
    "enter_valid": "Enter valid numbers",
    "import_failed": "Import Failed",
    "no_valid": "No valid data",
    "error": "Error",
    "not_found": "RobustRegressor.exe not found.\nCompile C++ first.",
    "cpp_error": "C++ Error: ",
    "call_failed": "Call failed: ",
    "deleted": "Deleted point #{}, remaining {}",
    "excluded": "Excluded",
    "included": "Included",
    "point_set": "Point #{} set to {}",
    "added": "Added point ({:.3f}, {:.3f})",
    "imported": "Imported {} points",
    "exported": "Exported to {}",
    "cleared": "All points cleared",
    "ai_detected": "AI Detected",
    "manually_excluded": "Manually Excluded",
    "normal": "Normal",
    "outlier": "AI Detected Outlier",
    "fitted": "Fitted",
    "green_normal": "green=normal",
    "red_outlier": "red=outlier",
    "orange_excl": "orange=excluded",
    "blue_sel": "blue=selected",
    "mouse_hint": "⚡ Mouse: Left(empty)=Add | Left(point)=Select | Right=Delete | Middle/Ctrl+Left=Toggle Exclude | Scroll=Zoom"
}


# ==================== 神经网络回归器 ====================

class NNRegressor:
    def __init__(self, dim=2, hidden=(128, 64, 32), epochs=600, lr=0.01):
        self.dim = dim
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.model = None
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None
        self.inliers = []
        self.outliers = []
        self.rmse = 0.0
        self.r2 = 0.0

    def fit(self, points, excluded):
        active = [p for i, p in enumerate(points) if not excluded[i]]
        n_active = len(active)
        if n_active < 5:
            return None

        if self.dim == 2:
            X = np.array([[p[0]] for p in active], dtype=np.float32)
            Y = np.array([[p[1]] for p in active], dtype=np.float32)
        else:
            X = np.array([[p[0], p[1]] for p in active], dtype=np.float32)
            Y = np.array([[p[2]] for p in active], dtype=np.float32)

        self.x_mean = X.mean(axis=0)
        self.x_std = X.std(axis=0) + 1e-8
        self.y_mean = float(Y.mean())
        self.y_std = float(Y.std()) + 1e-8

        Xn = (X - self.x_mean) / self.x_std
        Yn = (Y - self.y_mean) / self.y_std

        X_t = torch.from_numpy(Xn)
        Y_t = torch.from_numpy(Yn)

        layers = []
        prev = X.shape[1]
        for h in self.hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        model = nn.Sequential(*layers)

        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=200, gamma=0.5)

        weights = torch.ones(n_active, dtype=torch.float32)
        for stage in range(3):
            for epoch in range(self.epochs // 3):
                model.train()
                pred = model(X_t)
                loss = (weights * (pred - Y_t) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            scheduler.step()

            with torch.no_grad():
                pred = model(X_t)
                residuals = torch.abs(pred - Y_t).numpy().flatten()
                med = np.median(residuals)
                mad = np.median(np.abs(residuals - med)) + 1e-10
                z = 0.6745 * (residuals - med) / mad
                weights = torch.from_numpy(np.where(z < 3.0, 1.0, 0.05).astype(np.float32))

        self.model = model

        if self.dim == 2:
            X_full = np.array([[p[0]] for p in points], dtype=np.float32)
            Y_full = np.array([p[1] for p in points], dtype=np.float32)
        else:
            X_full = np.array([[p[0], p[1]] for p in points], dtype=np.float32)
            Y_full = np.array([p[2] for p in points], dtype=np.float32)

        X_full_n = (X_full - self.x_mean) / self.x_std
        with torch.no_grad():
            pred_full = model(torch.from_numpy(X_full_n)).numpy().flatten()
        pred_full = pred_full * self.y_std + self.y_mean

        residuals = np.abs(pred_full - Y_full)
        med = np.median(residuals)
        mad = np.median(np.abs(residuals - med)) + 1e-10
        z_scores = 0.6745 * (residuals - med) / mad

        self.inliers = [i for i, z in enumerate(z_scores) if z < 3.0]
        self.outliers = [i for i, z in enumerate(z_scores) if z >= 3.0]

        if len(self.inliers) > 0:
            y_in = Y_full[self.inliers]
            p_in = pred_full[self.inliers]
            self.rmse = np.sqrt(np.mean((y_in - p_in) ** 2))
            ss_res = np.sum((y_in - p_in) ** 2)
            ss_tot = np.sum((y_in - np.mean(y_in)) ** 2)
            self.r2 = 1.0 - ss_res / (ss_tot + 1e-10)
        else:
            self.rmse = 0.0
            self.r2 = 0.0

        return self

    def predict(self, X_query):
        if self.model is None:
            return None
        Xn = (X_query - self.x_mean) / self.x_std
        X_t = torch.from_numpy(Xn.astype(np.float32))
        with torch.no_grad():
            pred = self.model(X_t).numpy().flatten()
        return pred * self.y_std + self.y_mean


# ==================== 符号回归器 (改进版) ====================

class SparseSymbolicRegressor:
    def __init__(self, dim=2, max_degree=3, alpha=0.01):
        self.dim = dim
        self.max_degree = max_degree
        self.alpha = alpha
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.inliers = []
        self.outliers = []
        self.rmse = 0.0
        self.r2 = 0.0
        self.is_fitted = False

    def build_features(self, X):
        n_vars = X.shape[1]
        features = []
        names = []
        vars_symbols = ['x', 'y'][:n_vars]

        # 1. 线性项
        for i in range(n_vars):
            features.append(X[:, i])
            names.append(vars_symbols[i])

        # 2. 多项式项
        for d in range(2, self.max_degree + 1):
            for i in range(n_vars):
                features.append(X[:, i] ** d)
                names.append(f"{vars_symbols[i]}^{d}")

        # 3. 交互项
        if n_vars == 2:
            features.append(X[:, 0] * X[:, 1])
            names.append('x*y')
            if self.max_degree >= 3:
                features.append((X[:, 0] ** 2) * X[:, 1])
                names.append('x^2*y')
                features.append(X[:, 0] * (X[:, 1] ** 2))
                names.append('x*y^2')

        # 4. 正弦/余弦
        for i in range(n_vars):
            features.append(np.sin(X[:, i]))
            names.append(f"sin({vars_symbols[i]})")
            features.append(np.cos(X[:, i]))
            names.append(f"cos({vars_symbols[i]})")

        # 5. 指数衰减
        for i in range(n_vars):
            features.append(np.exp(-X[:, i] ** 2))
            names.append(f"exp(-{vars_symbols[i]}^2)")
            features.append(np.exp(-np.abs(X[:, i])))
            names.append(f"exp(-|{vars_symbols[i]}|)")

        # 6. 对数
        for i in range(n_vars):
            features.append(np.log(np.abs(X[:, i]) + 1e-8))
            names.append(f"log(|{vars_symbols[i]}|+1)")

        # 7. 平方根
        for i in range(n_vars):
            features.append(np.sqrt(np.abs(X[:, i]) + 1e-8))
            names.append(f"sqrt(|{vars_symbols[i]}|)")

        return np.column_stack(features), names

    def fit(self, points, excluded):
        active = [p for i, p in enumerate(points) if not excluded[i]]
        n_active = len(active)
        if n_active < 5:
            return None

        if self.dim == 2:
            X = np.array([[p[0]] for p in active], dtype=np.float64)
            Y = np.array([p[1] for p in active], dtype=np.float64)
        else:
            X = np.array([[p[0], p[1]] for p in active], dtype=np.float64)
            Y = np.array([p[2] for p in active], dtype=np.float64)

        F, self.feature_names = self.build_features(X)
        F_scaled = self.scaler.fit_transform(F)

        try:
            self.model = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0], cv=min(5, n_active))
            self.model.fit(F_scaled, Y)
        except Exception:
            self.model = Lasso(alpha=self.alpha, max_iter=20000, random_state=42)
            self.model.fit(F_scaled, Y)

        pred = self.model.predict(F_scaled)
        residuals = np.abs(pred - Y)
        med = np.median(residuals)
        mad = np.median(np.abs(residuals - med)) + 1e-10
        z_scores = 0.6745 * (residuals - med) / mad

        active_indices = [i for i, ex in enumerate(excluded) if not ex]
        self.inliers = [active_indices[i] for i, z in enumerate(z_scores) if z < 3.0]
        self.outliers = [active_indices[i] for i, z in enumerate(z_scores) if z >= 3.0]

        if self.dim == 2:
            X_full = np.array([[p[0]] for p in points], dtype=np.float64)
            Y_full = np.array([p[1] for p in points], dtype=np.float64)
        else:
            X_full = np.array([[p[0], p[1]] for p in points], dtype=np.float64)
            Y_full = np.array([p[2] for p in points], dtype=np.float64)

        F_full, _ = self.build_features(X_full)
        F_full_scaled = self.scaler.transform(F_full)
        pred_full = self.model.predict(F_full_scaled)

        self.rmse = np.sqrt(np.mean((pred_full - Y_full) ** 2))
        ss_res = np.sum((pred_full - Y_full) ** 2)
        ss_tot = np.sum((Y_full - np.mean(Y_full)) ** 2)
        self.r2 = 1.0 - ss_res / (ss_tot + 1e-10)

        self.is_fitted = True
        return self

    def predict(self, X_query):
        if self.model is None:
            return None
        F, _ = self.build_features(X_query)
        F_scaled = self.scaler.transform(F)
        return self.model.predict(F_scaled)

    def equation_str(self):
        if self.model is None:
            return "N/A"
        
        if hasattr(self.model, 'coef_'):
            coefs = self.model.coef_
            intercept = self.model.intercept_
        else:
            return "Complex model (cannot display equation)"

        terms = []
        if abs(intercept) > 1e-6:
            terms.append(f"{intercept:.4f}")
        
        for name, c in zip(self.feature_names, coefs):
            if abs(c) > 1e-4:
                sign = '+' if c >= 0 else '-'
                terms.append(f"{sign} {abs(c):.4f}*{name}")
        
        if not terms:
            return f"{'y' if self.dim == 2 else 'z'} = 0"
        
        eq = " ".join(terms)
        if eq.startswith('+'):
            eq = eq[2:]
        
        if len(eq) > 200:
            eq = eq[:197] + "..."
        
        target = "y" if self.dim == 2 else "z"
        return f"{target} = {eq}"


# ==================== 主 UI ====================

class RobustRegressorUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.t = UI_TEXTS
        self.setWindowTitle(self.t["title"])
        self.setGeometry(100, 100, 1500, 950)

        self.points = []
        self.excluded = []
        self.selected_idx = -1
        self.dim = 2
        self.cpp_result = None
        
        # 保存当前视图范围
        self.saved_xlim = None
        self.saved_ylim = None
        self.saved_zlim = None
        
        # 标志位，防止递归刷新
        self._updating_grid = False

        self.init_ui()
        self.refresh_plot()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_h = QHBoxLayout(central)
        main_h.setContentsMargins(2, 2, 2, 2)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_h.addWidget(splitter)

        # ==================== 左侧面板 ====================
        left_panel = QWidget()
        left_panel.setMaximumWidth(450)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(8)
        left_layout.setContentsMargins(5, 5, 5, 5)

        # 文件操作
        self.file_group = QGroupBox(self.t["file_ops"])
        fl = QVBoxLayout()
        self.btn_import = QPushButton(self.t["import_csv"])
        self.btn_import.clicked.connect(self.import_csv)
        self.btn_export = QPushButton(self.t["export_result"])
        self.btn_export.clicked.connect(self.export_result)
        self.btn_clear = QPushButton(self.t["clear_all"])
        self.btn_clear.clicked.connect(self.clear_all)
        fl.addWidget(self.btn_import)
        fl.addWidget(self.btn_export)
        fl.addWidget(self.btn_clear)
        self.file_group.setLayout(fl)
        left_layout.addWidget(self.file_group)

        # 模型设置
        self.model_group = QGroupBox(self.t["model_settings"])
        ml = QVBoxLayout()

        mode_h = QHBoxLayout()
        self.lbl_mode = QLabel(self.t["mode"])
        mode_h.addWidget(self.lbl_mode)
        self.combo_mode = QComboBox()
        self.combo_mode.addItems([self.t["classic_ransac"], self.t["neural_net"], self.t["symbolic_lasso"]])
        self.combo_mode.currentTextChanged.connect(self.on_mode_changed)
        mode_h.addWidget(self.combo_mode)
        ml.addLayout(mode_h)

        dim_h = QHBoxLayout()
        self.lbl_dim = QLabel(self.t["dimension"])
        dim_h.addWidget(self.lbl_dim)
        self.combo_dim = QComboBox()
        self.combo_dim.addItems([self.t["2d"], self.t["3d"]])
        self.combo_dim.currentTextChanged.connect(self.on_dim_changed)
        dim_h.addWidget(self.combo_dim)
        ml.addLayout(dim_h)

        deg_h = QHBoxLayout()
        self.lbl_degree = QLabel(self.t["poly_degree"])
        deg_h.addWidget(self.lbl_degree)
        self.spin_degree = QSpinBox()
        self.spin_degree.setRange(0, 5)
        self.spin_degree.setValue(0)
        self.spin_degree.setToolTip(self.t["auto_classic"])
        deg_h.addWidget(self.spin_degree)
        ml.addLayout(deg_h)

        thr_h = QHBoxLayout()
        self.lbl_thresh = QLabel(self.t["ransac_thresh"])
        thr_h.addWidget(self.lbl_thresh)
        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(-1, 10000)
        self.spin_threshold.setValue(-1)
        self.spin_threshold.setDecimals(4)
        self.spin_threshold.setToolTip(self.t["auto_thresh"])
        thr_h.addWidget(self.spin_threshold)
        ml.addLayout(thr_h)

        iter_h = QHBoxLayout()
        self.lbl_iter = QLabel(self.t["max_iter"])
        iter_h.addWidget(self.lbl_iter)
        self.spin_iter = QSpinBox()
        self.spin_iter.setRange(100, 50000)
        self.spin_iter.setValue(2000)
        self.spin_iter.setSingleStep(500)
        iter_h.addWidget(self.spin_iter)
        ml.addLayout(iter_h)

        self.chk_realtime = QCheckBox(self.t["realtime"])
        self.chk_realtime.setChecked(True)
        ml.addWidget(self.chk_realtime)

        self.btn_fit = QPushButton(self.t["run_fit"])
        self.btn_fit.clicked.connect(self.run_regression)
        ml.addWidget(self.btn_fit)

        self.model_group.setLayout(ml)
        left_layout.addWidget(self.model_group)

        # 网格设置
        self.grid_group = QGroupBox("Grid Settings")
        gl = QVBoxLayout()

        # 网格密度滑块
        grid_h = QHBoxLayout()
        self.lbl_grid = QLabel("Grid Density:")
        self.grid_slider = QSlider(Qt.Orientation.Horizontal)
        self.grid_slider.setRange(2, 20)
        self.grid_slider.setValue(8)
        self.grid_slider.valueChanged.connect(self.on_grid_density_changed)
        self.grid_density_label = QLabel("8")
        grid_h.addWidget(self.lbl_grid)
        grid_h.addWidget(self.grid_slider)
        grid_h.addWidget(self.grid_density_label)
        gl.addLayout(grid_h)

        # 网格样式选择
        style_h = QHBoxLayout()
        self.lbl_grid_style = QLabel("Grid Style:")
        self.combo_grid_style = QComboBox()
        self.combo_grid_style.addItems(["Both", "Major", "None"])
        self.combo_grid_style.setCurrentText("Both")
        self.combo_grid_style.currentTextChanged.connect(self.on_grid_style_changed)
        style_h.addWidget(self.lbl_grid_style)
        style_h.addWidget(self.combo_grid_style)
        gl.addLayout(style_h)

        # 网格透明度
        alpha_h = QHBoxLayout()
        self.lbl_grid_alpha = QLabel("Grid Alpha:")
        self.grid_alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.grid_alpha_slider.setRange(10, 100)
        self.grid_alpha_slider.setValue(30)
        self.grid_alpha_slider.valueChanged.connect(self.on_grid_alpha_changed)
        self.grid_alpha_label = QLabel("0.30")
        alpha_h.addWidget(self.lbl_grid_alpha)
        alpha_h.addWidget(self.grid_alpha_slider)
        alpha_h.addWidget(self.grid_alpha_label)
        gl.addLayout(alpha_h)

        self.grid_group.setLayout(gl)
        left_layout.addWidget(self.grid_group)

        # 手动添加点
        self.add_group = QGroupBox(self.t["add_point"])
        al = QVBoxLayout()
        self.edit_x = QLineEdit()
        self.edit_x.setPlaceholderText(self.t["x"])
        self.edit_y = QLineEdit()
        self.edit_y.setPlaceholderText(self.t["y"])
        self.edit_z = QLineEdit()
        self.edit_z.setPlaceholderText(self.t["z_3d"])
        self.edit_z.setEnabled(False)
        al.addWidget(self.edit_x)
        al.addWidget(self.edit_y)
        al.addWidget(self.edit_z)
        self.btn_add = QPushButton(self.t["add_point"])
        self.btn_add.clicked.connect(self.manual_add_point)
        al.addWidget(self.btn_add)
        self.add_group.setLayout(al)
        left_layout.addWidget(self.add_group)

        # 统计信息
        self.info_group = QGroupBox(self.t["fit_info"])
        il = QVBoxLayout()
        self.lbl_equation = QLabel(f"{self.t['equation']} {self.t['na']}")
        self.lbl_equation.setWordWrap(True)
        self.lbl_stats = QLabel(self.t["r2_rmse"])
        self.lbl_counts = QLabel(self.t["counts"])
        il.addWidget(self.lbl_equation)
        il.addWidget(self.lbl_stats)
        il.addWidget(self.lbl_counts)
        self.info_group.setLayout(il)
        left_layout.addWidget(self.info_group)

        # 异常点列表
        self.out_group = QGroupBox(self.t["outliers_list"])
        ol = QVBoxLayout()
        self.table_outliers = QTableWidget()
        self.table_outliers.setColumnCount(3)
        self.table_outliers.setHorizontalHeaderLabels([self.t["index"], self.t["coords"], self.t["reason"]])
        self.table_outliers.horizontalHeader().setStretchLastSection(True)
        self.table_outliers.setMaximumHeight(200)
        ol.addWidget(self.table_outliers)
        self.out_group.setLayout(ol)
        left_layout.addWidget(self.out_group)

        left_layout.addStretch()
        splitter.addWidget(left_panel)

        # ==================== 右侧绘图区 ====================
        right_panel = QWidget()
        rl = QVBoxLayout(right_panel)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(2)
        
        # 工具栏
        self.fig = Figure(figsize=(8, 6), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        rl.addWidget(self.toolbar)
        
        # 绘图区域
        rl.addWidget(self.canvas)
        
        # 鼠标操作提示栏
        hint_frame = QFrame()
        hint_frame.setFrameShape(QFrame.Shape.StyledPanel)
        hint_frame.setStyleSheet("QFrame { background-color: #f0f0f0; border: 1px solid #cccccc; border-radius: 3px; }")
        hint_layout = QHBoxLayout(hint_frame)
        hint_layout.setContentsMargins(8, 4, 8, 4)
        
        self.hint_label = QLabel(self.t["mouse_hint"])
        self.hint_label.setStyleSheet("color: #555555; font-size: 11px; font-family: monospace;")
        self.hint_label.setWordWrap(True)
        hint_layout.addWidget(self.hint_label)
        
        rl.addWidget(hint_frame)
        
        splitter.addWidget(right_panel)

        splitter.setSizes([400, 1100])
        self.statusBar().showMessage(self.t["ready"])

        # 连接画布事件
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)

    # ========== 网格设置方法 ==========
    def on_grid_density_changed(self, value):
        if self._updating_grid:
            return
        self.grid_density_label.setText(str(value))
        self.refresh_plot()

    def on_grid_style_changed(self, style):
        if self._updating_grid:
            return
        self.refresh_plot()

    def on_grid_alpha_changed(self, value):
        if self._updating_grid:
            return
        alpha = value / 100.0
        self.grid_alpha_label.setText(f"{alpha:.2f}")
        self.refresh_plot()

    def apply_grid_settings(self, ax):
        """应用网格设置到坐标轴 - 使用 Locator 而不是固定刻度，避免卡顿"""
        style = self.combo_grid_style.currentText()
        alpha = self.grid_alpha_slider.value() / 100.0
        density = self.grid_slider.value()
        
        if style == "None":
            ax.grid(False)
            return
        
        # 使用 MaxNLocator 自动选择合适的刻度数量，而不是固定位置
        # 这样缩放时不会重新计算大量刻度，避免卡顿
        ax.xaxis.set_major_locator(MaxNLocator(density, prune='both'))
        ax.yaxis.set_major_locator(MaxNLocator(density, prune='both'))
        
        # 设置次刻度（自动）
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        
        # 设置网格
        if style == "Both":
            ax.grid(True, which='major', alpha=alpha, linestyle='--', linewidth=0.8)
            ax.grid(True, which='minor', alpha=alpha * 0.5, linestyle=':', linewidth=0.5)
        else:  # "Major" only
            ax.grid(True, which='major', alpha=alpha, linestyle='--', linewidth=0.8)
            ax.grid(False, which='minor')
        
        ax.set_axisbelow(True)

    def save_current_view(self):
        """保存当前视图范围"""
        if self.dim == 2 and self.fig.axes:
            ax = self.fig.axes[0]
            self.saved_xlim = ax.get_xlim()
            self.saved_ylim = ax.get_ylim()
        elif self.dim == 3 and self.fig.axes:
            ax = self.fig.axes[0]
            self.saved_xlim = ax.get_xlim()
            self.saved_ylim = ax.get_ylim()
            self.saved_zlim = ax.get_zlim()

    def restore_view(self):
        """恢复之前保存的视图范围"""
        if self.dim == 2 and self.saved_xlim is not None and self.fig.axes:
            ax = self.fig.axes[0]
            ax.set_xlim(self.saved_xlim)
            ax.set_ylim(self.saved_ylim)
        elif self.dim == 3 and self.saved_xlim is not None and self.fig.axes:
            ax = self.fig.axes[0]
            ax.set_xlim(self.saved_xlim)
            ax.set_ylim(self.saved_ylim)
            if self.saved_zlim is not None:
                ax.set_zlim(self.saved_zlim)

    def clear_all(self):
        self.points = []
        self.excluded = []
        self.selected_idx = -1
        self.cpp_result = None
        self.inlier_indices = []
        self.outlier_indices = []
        self.equation_str = ""
        self.rmse = 0.0
        self.r2 = 0.0
        self.saved_xlim = None
        self.saved_ylim = None
        self.statusBar().showMessage(self.t["cleared"])
        self.update_info()
        self.refresh_plot()

    def import_csv(self):
        fn, _ = QFileDialog.getOpenFileName(self, self.t["import_csv"], "", "CSV (*.csv)")
        if not fn:
            return
        try:
            pts = []
            with open(fn, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row or row[0].startswith('#') or row[0].strip() == '':
                        continue
                    vals = [float(v.strip()) for v in row if v.strip()]
                    if len(vals) == self.dim:
                        pts.append(vals)
                    elif len(vals) == 2 and self.dim == 3:
                        pts.append([vals[0], vals[1], 0.0])
                    elif len(vals) == 3 and self.dim == 2:
                        pts.append([vals[0], vals[1]])
            if pts:
                self.points = pts
                self.excluded = [False] * len(self.points)
                self.selected_idx = -1
                self.statusBar().showMessage(self.t["imported"].format(len(pts)))
                self.run_regression()
            else:
                QMessageBox.warning(self, self.t["import_failed"], self.t["no_valid"])
        except Exception as e:
            QMessageBox.critical(self, self.t["error"], str(e))

    def export_result(self):
        fn, _ = QFileDialog.getSaveFileName(self, self.t["export_result"], "result.txt", "Text (*.txt)")
        if not fn:
            return
        try:
            with open(fn, 'w', encoding='utf-8') as f:
                f.write(f"Mode: {self.combo_mode.currentText()}\n")
                f.write(f"Dimension: {self.dim}D\n")
                f.write(f"Equation: {getattr(self, 'equation_str', 'N/A')}\n")
                f.write(f"R2: {getattr(self, 'r2', 0):.6f}\n")
                f.write(f"RMSE: {getattr(self, 'rmse', 0):.6f}\n")
                f.write(f"Total: {len(self.points)}\n")
                f.write(f"Inliers: {len(getattr(self, 'inlier_indices', []))}\n")
                f.write(f"Outliers: {len(getattr(self, 'outlier_indices', []))}\n")
                f.write(f"Excluded: {sum(self.excluded)}\n\n")
                f.write("Data Points:\n")
                for i, p in enumerate(self.points):
                    status = self.t["normal"]
                    if self.excluded[i]:
                        status = self.t["manually_excluded"]
                    elif i in getattr(self, 'outlier_indices', []):
                        status = self.t["outlier"]
                    f.write(f"{i}: {', '.join(f'{v:.6f}' for v in p)} [{status}]\n")
            self.statusBar().showMessage(self.t["exported"].format(fn))
        except Exception as e:
            QMessageBox.critical(self, self.t["error"], str(e))

    def on_dim_changed(self, text):
        new_dim = 2 if text == self.t["2d"] else 3
        if new_dim == self.dim:
            return
        self.dim = new_dim
        self.points = []
        self.excluded = []
        self.selected_idx = -1
        self.edit_z.setEnabled(self.dim == 3)
        self.saved_xlim = None
        self.saved_ylim = None
        self.saved_zlim = None
        self.refresh_plot()

    def on_mode_changed(self, text):
        is_ransac = (text == self.t["classic_ransac"])
        self.spin_degree.setEnabled(is_ransac)
        self.spin_threshold.setEnabled(is_ransac)
        self.spin_iter.setEnabled(is_ransac)
        if self.points and len(self.points) > 0:
            self.run_regression()

    def manual_add_point(self):
        """手动添加点（通过输入框）"""
        try:
            x = float(self.edit_x.text())
            y = float(self.edit_y.text())
            if self.dim == 3:
                z = float(self.edit_z.text())
                self.points.append([x, y, z])
            else:
                self.points.append([x, y])
            self.excluded.append(False)
            self.edit_x.clear()
            self.edit_y.clear()
            self.edit_z.clear()
            self.statusBar().showMessage(self.t["added"].format(x, y))
            if self.chk_realtime.isChecked():
                self.run_regression()
            else:
                self.refresh_plot()
        except ValueError:
            QMessageBox.warning(self, self.t["input_error"], self.t["enter_valid"])

    def on_canvas_click(self, event):
        """画布点击事件"""
        # 如果工具栏处于平移/缩放模式，忽略点击
        if self.toolbar.mode:
            return
        if event.inaxes is None:
            return

        # 获取点击位置
        x, y = event.xdata, event.ydata
        
        # 如果没有点，直接添加
        if not self.points:
            if event.button == 1:  # 左键
                self.add_point_at(x, y)
            return

        # 查找最近的点的距离（像素单位）
        ax = event.inaxes
        # 获取坐标轴范围
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        # 获取画布尺寸
        bbox = ax.get_window_extent()
        width = bbox.width
        height = bbox.height
        
        # 计算像素到坐标的映射
        x_per_pixel = (xlim[1] - xlim[0]) / width
        y_per_pixel = (ylim[1] - ylim[0]) / height
        
        best_idx = -1
        best_dist = float('inf')
        
        for i, p in enumerate(self.points):
            # 计算像素距离
            dx_px = (p[0] - x) / x_per_pixel
            dy_px = (p[1] - y) / y_per_pixel
            dist_px = np.sqrt(dx_px * dx_px + dy_px * dy_px)
            
            if dist_px < best_dist:
                best_dist = dist_px
                best_idx = i
        
        # 阈值15像素
        if best_dist > 15:
            # 没有点到任何点，添加新点
            if event.button == 1:
                self.add_point_at(x, y)
            return
        
        # 有点被选中
        if event.button == 1:  # 左键 - 选中
            self.selected_idx = best_idx
            self.refresh_plot()
        elif event.button == 3:  # 右键 - 删除
            del self.points[best_idx]
            del self.excluded[best_idx]
            if self.selected_idx == best_idx:
                self.selected_idx = -1
            elif self.selected_idx > best_idx:
                self.selected_idx -= 1
            self.statusBar().showMessage(self.t["deleted"].format(best_idx, len(self.points)))
            if self.chk_realtime.isChecked():
                self.run_regression()
            else:
                self.refresh_plot()
        elif event.button == 2 or (event.button == 1 and event.key == 'control'):  # 中键或 Ctrl+左键 - 排除/包含
            self.excluded[best_idx] = not self.excluded[best_idx]
            state = self.t["excluded"] if self.excluded[best_idx] else self.t["included"]
            self.statusBar().showMessage(self.t["point_set"].format(best_idx, state))
            if self.chk_realtime.isChecked():
                self.run_regression()
            else:
                self.refresh_plot()

    def add_point_at(self, x, y):
        """在指定坐标添加点"""
        # 保存当前视图
        self.save_current_view()
        
        if self.dim == 2:
            self.points.append([x, y])
        else:
            # 3D模式需要输入Z坐标
            from PyQt6.QtWidgets import QInputDialog
            z, ok = QInputDialog.getDouble(self, "Enter Z", "z:", 0, -1e9, 1e9, 4)
            if not ok:
                return
            self.points.append([x, y, z])
        
        self.excluded.append(False)
        self.statusBar().showMessage(self.t["added"].format(x, y))
        
        # 运行回归或刷新
        if self.chk_realtime.isChecked():
            self.run_regression()
        else:
            self.refresh_plot()
        
        # 恢复视图
        self.restore_view()
        self.canvas.draw_idle()

    def find_exe(self):
        candidates = [
            "RobustRegressor.exe",
            "robust_regressor.exe",
            "x64/Release/RobustRegressor.exe",
            "x64/Debug/RobustRegressor.exe",
            "Release/RobustRegressor.exe",
            "Debug/RobustRegressor.exe",
        ]
        for c in candidates:
            if os.path.exists(c):
                return os.path.abspath(c)
        return None

    def run_ransac(self):
        active_indices = [i for i, ex in enumerate(self.excluded) if not ex]
        if not active_indices:
            self.refresh_plot()
            return

        tmp_input = "_tmp_input.csv"
        with open(tmp_input, 'w', newline='') as f:
            writer = csv.writer(f)
            for i in active_indices:
                writer.writerow([f"{v:.8f}" for v in self.points[i]])

        exe = self.find_exe()
        if not exe:
            QMessageBox.critical(self, self.t["error"], self.t["not_found"])
            return

        degree = self.spin_degree.value()
        threshold = self.spin_threshold.value()
        max_iter = self.spin_iter.value()
        tmp_output = "_tmp_result.csv"

        cmd = [exe, tmp_input, str(self.dim), str(degree), str(threshold), str(max_iter), tmp_output]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if r.returncode != 0:
                self.statusBar().showMessage(self.t["cpp_error"] + r.stderr[:100])
                return
        except Exception as e:
            self.statusBar().showMessage(self.t["call_failed"] + str(e))
            return

        if not os.path.exists(tmp_output):
            return
        with open(tmp_output, 'r', encoding='utf-8') as f:
            self.cpp_result = f.read()

        self.parse_cpp_result(tmp_output, active_indices)

        self.inlier_indices = getattr(self, 'inlier_indices', [])
        self.outlier_indices = getattr(self, 'outlier_indices', [])
        self.equation_str = getattr(self, 'equation_str', '')

        self.refresh_plot()
        self.update_info()

        for f in [tmp_input, tmp_output]:
            if os.path.exists(f):
                os.remove(f)

    def parse_cpp_result(self, fn, active_indices):
        self.inlier_indices = []
        self.outlier_indices = []
        self.equation_str = ""
        self.rmse = 0.0
        self.r2 = 0.0

        section = None
        with open(fn, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or all(v.strip() == '' for v in row):
                    continue

                line = ','.join(row).strip()

                if line.startswith('# Equation:'):
                    self.equation_str = line.replace('# Equation:', '').strip()
                elif line.startswith('# RMSE:'):
                    self.rmse = float(line.split(':')[1].strip())
                elif line.startswith('# R_squared:'):
                    self.r2 = float(line.split(':')[1].strip())
                elif line == '# Cleaned Data (Inliers):':
                    section = 'inliers'
                elif line == '# Outliers:':
                    section = 'outliers'
                elif line.startswith('#') or line.startswith('x,'):
                    continue
                else:
                    vals = [v.strip() for v in row if v.strip()]
                    if len(vals) < self.dim:
                        continue
                    try:
                        coords = [float(v) for v in vals[:self.dim]]
                    except ValueError:
                        continue

                    idx = self.find_point_index(coords)
                    if idx >= 0:
                        if section == 'inliers':
                            self.inlier_indices.append(idx)
                        elif section == 'outliers':
                            self.outlier_indices.append(idx)

    def find_point_index(self, coords):
        for i, p in enumerate(self.points):
            if all(abs(p[j] - coords[j]) < 1e-4 for j in range(self.dim)):
                return i
        return -1

    def run_regression(self):
        # 保存当前视图
        self.save_current_view()
        
        if len(self.points) < (3 if self.dim == 3 else 2):
            self.refresh_plot()
            self.restore_view()
            return

        mode = self.combo_mode.currentText()
        active_indices = [i for i, ex in enumerate(self.excluded) if not ex]
        if not active_indices:
            self.refresh_plot()
            self.restore_view()
            return

        if mode == self.t["classic_ransac"]:
            self.run_ransac()
        elif mode == self.t["neural_net"]:
            self.run_neural()
        elif mode == self.t["symbolic_lasso"]:
            self.run_symbolic()
        
        # 恢复视图
        self.restore_view()

    def run_neural(self):
        self.nn_regressor = NNRegressor(dim=self.dim)
        self.nn_regressor.fit(self.points, self.excluded)
        self.inlier_indices = self.nn_regressor.inliers
        self.outlier_indices = self.nn_regressor.outliers
        self.equation_str = "Neural Network MLP"
        self.rmse = self.nn_regressor.rmse
        self.r2 = self.nn_regressor.r2
        self.refresh_plot()
        self.update_info()

    def run_symbolic(self):
        self.sym_regressor = SparseSymbolicRegressor(dim=self.dim, max_degree=3, alpha=0.005)
        self.sym_regressor.fit(self.points, self.excluded)
        self.inlier_indices = self.sym_regressor.inliers
        self.outlier_indices = self.sym_regressor.outliers
        self.equation_str = self.sym_regressor.equation_str()
        self.rmse = self.sym_regressor.rmse
        self.r2 = self.sym_regressor.r2
        self.refresh_plot()
        self.update_info()

    def update_info(self):
        eq = getattr(self, 'equation_str', self.t["na"])
        self.lbl_equation.setText(f"{self.t['equation']} {eq}")
        self.lbl_stats.setText(f"R2: {getattr(self, 'r2', 0):.6f} | RMSE: {getattr(self, 'rmse', 0):.6f}")
        n_total = len(self.points)
        n_excl = sum(self.excluded)
        n_out = len(getattr(self, 'outlier_indices', []))
        n_in = n_total - n_excl - n_out
        self.lbl_counts.setText(
            f"{self.t['green_normal']} | {self.t['red_outlier']} | {self.t['orange_excl']} | {self.t['blue_sel']}"
            + f"\nTotal: {n_total} | Inliers: {n_in} | Outliers: {n_out} | Excluded: {n_excl}"
        )

        self.table_outliers.setRowCount(0)
        for idx in getattr(self, 'outlier_indices', []):
            row = self.table_outliers.rowCount()
            self.table_outliers.insertRow(row)
            self.table_outliers.setItem(row, 0, QTableWidgetItem(str(idx)))
            coord_str = ", ".join(f"{v:.3f}" for v in self.points[idx])
            self.table_outliers.setItem(row, 1, QTableWidgetItem(coord_str))
            self.table_outliers.setItem(row, 2, QTableWidgetItem(self.t["ai_detected"]))
        for i, ex in enumerate(self.excluded):
            if ex:
                row = self.table_outliers.rowCount()
                self.table_outliers.insertRow(row)
                self.table_outliers.setItem(row, 0, QTableWidgetItem(str(i)))
                coord_str = ", ".join(f"{v:.3f}" for v in self.points[i])
                self.table_outliers.setItem(row, 1, QTableWidgetItem(coord_str))
                self.table_outliers.setItem(row, 2, QTableWidgetItem(self.t["manually_excluded"]))

    def refresh_plot(self):
        # 更新标签显示
        self.grid_density_label.setText(str(self.grid_slider.value()))
        alpha = self.grid_alpha_slider.value() / 100.0
        self.grid_alpha_label.setText(f"{alpha:.2f}")
        
        # 清除并重绘
        self.fig.clf()
        if self.dim == 2:
            self.plot_2d()
        else:
            self.plot_3d()
        self.canvas.draw_idle()

    def plot_2d(self):
        ax = self.fig.add_subplot(111)
        if not self.points:
            ax.set_title("2D - " + self.t["ready"])
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            self.apply_grid_settings(ax)
            return

        pts = np.array(self.points)
        x_all, y_all = pts[:, 0], pts[:, 1]

        colors = []
        for i in range(len(self.points)):
            if i == self.selected_idx:
                colors.append('blue')
            elif self.excluded[i]:
                colors.append('orange')
            elif i in getattr(self, 'outlier_indices', []):
                colors.append('red')
            else:
                colors.append('green')

        ax.scatter(x_all, y_all, c=colors, s=80, alpha=0.8,
                   edgecolors='black', linewidth=0.5, zorder=3)

        mode = self.combo_mode.currentText()
        if len(self.points) > 1:
            x_min, x_max = x_all.min(), x_all.max()
            pad = (x_max - x_min) * 0.1 + 1e-6
            x_line = np.linspace(x_min - pad, x_max + pad, 300)
            X_query = np.array([[xi] for xi in x_line], dtype=np.float32)
            y_line = None

            if mode == self.t["classic_ransac"] and self.cpp_result:
                y_line = self.eval_equation_2d(x_line)
            elif mode == self.t["neural_net"] and hasattr(self, 'nn_regressor') and self.nn_regressor.model:
                y_line = self.nn_regressor.predict(X_query)
            elif mode == self.t["symbolic_lasso"] and hasattr(self, 'sym_regressor') and self.sym_regressor.is_fitted:
                y_line = self.sym_regressor.predict(X_query)

            if y_line is not None:
                ax.plot(x_line, y_line, 'r-', lw=2.5, label=self.t["fitted"], zorder=2)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        title = f'2D {mode}\n({self.t["green_normal"]} {self.t["red_outlier"]} {self.t["orange_excl"]} {self.t["blue_sel"]})'
        ax.set_title(title, fontsize=11)
        
        self.apply_grid_settings(ax)
        
        ax.legend(loc='best')
        ax.ticklabel_format(style='plain', useOffset=False)
        
        # 恢复保存的视图范围
        if self.saved_xlim is not None:
            ax.set_xlim(self.saved_xlim)
            ax.set_ylim(self.saved_ylim)

    def plot_3d(self):
        from mpl_toolkits.mplot3d import Axes3D
        ax = self.fig.add_subplot(111, projection='3d')
        if not self.points:
            ax.set_title("3D - " + self.t["ready"])
            style = self.combo_grid_style.currentText()
            if style != "None":
                alpha = self.grid_alpha_slider.value() / 100.0
                ax.grid(True, alpha=alpha)
            else:
                ax.grid(False)
            return

        pts = np.array(self.points)
        x_all, y_all, z_all = pts[:, 0], pts[:, 1], pts[:, 2]

        colors = []
        for i in range(len(self.points)):
            if i == self.selected_idx:
                colors.append('blue')
            elif self.excluded[i]:
                colors.append('orange')
            elif i in getattr(self, 'outlier_indices', []):
                colors.append('red')
            else:
                colors.append('green')

        ax.scatter(x_all, y_all, z_all, c=colors, s=60, alpha=0.8,
                   edgecolors='black', linewidth=0.5)

        mode = self.combo_mode.currentText()
        if len(self.points) >= 3:
            try:
                x_min, x_max = x_all.min(), x_all.max()
                y_min, y_max = y_all.min(), y_all.max()
                pad_x = (x_max - x_min) * 0.1 + 1e-6
                pad_y = (y_max - y_min) * 0.1 + 1e-6
                xx, yy = np.meshgrid(
                    np.linspace(x_min - pad_x, x_max + pad_x, 25),
                    np.linspace(y_min - pad_y, y_max + pad_y, 25)
                )
                X_grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
                zz = None

                if mode == self.t["classic_ransac"] and self.cpp_result:
                    if len(getattr(self, 'inlier_indices', [])) >= 3:
                        inlier_pts = np.array([self.points[i] for i in self.inlier_indices])
                        A = np.c_[inlier_pts[:, 0], inlier_pts[:, 1], np.ones(len(inlier_pts))]
                        coeff, _, _, _ = np.linalg.lstsq(A, inlier_pts[:, 2], rcond=None)
                        a, b, c = coeff
                        zz = a * xx + b * yy + c
                elif mode == self.t["neural_net"] and hasattr(self, 'nn_regressor') and self.nn_regressor.model:
                    zz = self.nn_regressor.predict(X_grid).reshape(xx.shape)
                elif mode == self.t["symbolic_lasso"] and hasattr(self, 'sym_regressor') and self.sym_regressor.is_fitted:
                    zz = self.sym_regressor.predict(X_grid).reshape(xx.shape)

                if zz is not None:
                    ax.plot_surface(xx, yy, zz, alpha=0.3, color='red', rstride=1, cstride=1)
            except Exception:
                pass

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'3D {mode} ({self.t["green_normal"]} {self.t["red_outlier"]} {self.t["orange_excl"]} {self.t["blue_sel"]})')
        
        style = self.combo_grid_style.currentText()
        if style != "None":
            alpha = self.grid_alpha_slider.value() / 100.0
            ax.grid(True, alpha=alpha)
        else:
            ax.grid(False)
        
        # 恢复保存的视图范围
        if self.saved_xlim is not None:
            ax.set_xlim(self.saved_xlim)
            ax.set_ylim(self.saved_ylim)
        if self.saved_zlim is not None:
            ax.set_zlim(self.saved_zlim)

    def eval_equation_2d(self, x_vals):
        if not getattr(self, 'equation_str', ''):
            return None
        try:
            import re
            eq = self.equation_str.replace(' ', '').replace('y=', '')
            coeffs = []
            tokens = re.split(r'(?=[+-])', eq)
            for tok in tokens:
                if not tok:
                    continue
                if '*x^' in tok:
                    c_str = tok.split('*x^')[0]
                    c = float(c_str) if c_str not in ('', '+', '-') else (1.0 if c_str != '-' else -1.0)
                    coeffs.append((float(c), int(tok.split('*x^')[1])))
                elif '*x' in tok:
                    c_str = tok.split('*x')[0]
                    c = float(c_str) if c_str not in ('', '+', '-') else (1.0 if c_str != '-' else -1.0)
                    coeffs.append((float(c), 1))
                else:
                    c = float(tok)
                    coeffs.append((float(c), 0))

            y = np.zeros_like(x_vals)
            for c, d in coeffs:
                y += c * (x_vals ** d)
            return y
        except Exception:
            if len(getattr(self, 'inlier_indices', [])) > 0:
                pts = np.array([self.points[i] for i in self.inlier_indices])
                deg = min(self.spin_degree.value(), len(pts) - 1)
                if deg < 1:
                    deg = 1
                z = np.polyfit(pts[:, 0], pts[:, 1], deg)
                p = np.poly1d(z)
                return p(x_vals)
            return None

    def on_scroll(self, event):
        """鼠标滚轮缩放"""
        ax = event.inaxes
        if ax is None:
            return

        # 获取当前范围
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # 缩放中心
        if event.xdata is None or event.ydata is None:
            x_center = (xlim[0] + xlim[1]) / 2
            y_center = (ylim[0] + ylim[1]) / 2
        else:
            x_center = event.xdata
            y_center = event.ydata

        # 计算缩放比例
        scale = 0.8 if event.button == 'up' else 1.25

        # 新的宽度和高度
        new_width = (xlim[1] - xlim[0]) * scale
        new_height = (ylim[1] - ylim[0]) * scale

        # 计算新的范围
        x_ratio = (x_center - xlim[0]) / (xlim[1] - xlim[0])
        y_ratio = (y_center - ylim[0]) / (ylim[1] - ylim[0])

        new_xlim = [x_center - new_width * x_ratio, x_center + new_width * (1 - x_ratio)]
        new_ylim = [y_center - new_height * y_ratio, y_center + new_height * (1 - y_ratio)]

        # 应用新范围
        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)

        # 更新保存的视图
        self.saved_xlim = new_xlim
        self.saved_ylim = new_ylim

        # 重绘
        self.canvas.draw_idle()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = RobustRegressorUI()
    window.show()
    sys.exit(app.exec())