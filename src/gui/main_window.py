import os
import sys
import json        # [분석 저장] 워크스페이스 저장/불러오기에 사용
import datetime   # [LOG] 로그 기록 시간 저장용

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap, QPalette, QColor
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from src.engine.data_engine import DataEngine
from src.engine.model_engine import ModelEngine


try:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

LIGHT_QSS = """
QMainWindow, QWidget { background-color: #F2F2F0; color: #1A1A1A; font-family: 'Malgun Gothic', sans-serif; }
QTabWidget::pane { border: 1px solid #D8D8D6; background: #FFFFFF; }
QTabBar { background: #EBEBEA; }
QTabBar::tab { background: #EBEBEA; color: #8A8A8A; padding: 6px 14px; border: 1px solid #D8D8D6; border-bottom: none; font-size: 11px; }
QTabBar::tab:selected { background: #FFFFFF; color: #1A1A1A; font-weight: bold; border-bottom: 2px solid #E56020; }
QTabBar::tab:hover { background: #FFFFFF; color: #1A1A1A; }
QGroupBox { border: 1px solid #D8D8D6; border-radius: 3px; margin-top: 8px; padding-top: 8px; font-weight: bold; font-size: 11px; color: #444444; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #8A8A8A; font-size: 10px; letter-spacing: 1px; }
QPushButton { background-color: #EBEBEA; color: #1A1A1A; border: 1px solid #D8D8D6; border-radius: 2px; padding: 5px 12px; font-size: 11px; }
QPushButton:hover { background-color: #D8D8D6; }
QPushButton:disabled { color: #8A8A8A; background: #F2F2F0; border-color: #EBEBEA; }
QComboBox { border: none; border-bottom: 2px solid #D8D8D6; background: transparent; padding: 4px 2px; font-size: 11px; color: #1A1A1A; }
QComboBox:focus { border-bottom-color: #E56020; }
QComboBox QAbstractItemView { background: #FFFFFF; border: 1px solid #D8D8D6; selection-background-color: #E56020; }
QTableWidget { border: 1px solid #D8D8D6; gridline-color: #EBEBEA; font-size: 11px; background: #FFFFFF; alternate-background-color: #F8F8F7; }
QTableWidget::item { background-color: #FFFFFF; color: #1A1A1A; }
QTableWidget::item:alternate { background-color: #F8F8F7; }
QTableWidget::item:selected { background: #E56020; color: white; }
QHeaderView::section { background: #EBEBEA; color: #8A8A8A; border: none; border-right: 1px solid #D8D8D6; border-bottom: 1px solid #D8D8D6; padding: 4px 8px; font-size: 10px; font-weight: 600; letter-spacing: 0.5px; }
QScrollBar:vertical { width: 6px; background: transparent; }
QScrollBar::handle:vertical { background: #D8D8D6; border-radius: 3px; min-height: 20px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal { height: 6px; background: transparent; }
QScrollBar::handle:horizontal { background: #D8D8D6; border-radius: 3px; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QLineEdit { border: none; border-bottom: 2px solid #D8D8D6; background: transparent; padding: 4px 2px; font-size: 11px; }
QLineEdit:focus { border-bottom-color: #E56020; }
QDoubleSpinBox, QSpinBox { border: none; border-bottom: 2px solid #D8D8D6; background: transparent; padding: 4px 2px; font-size: 11px; }
QSplitter::handle { background: #D8D8D6; }
QDialog { background: #FFFFFF; }
QMessageBox { background: #FFFFFF; }
"""

DARK_QSS = """
QMainWindow, QWidget { background-color: #2B2B2B; color: #D8D8D8; font-family: 'Malgun Gothic', sans-serif; }
QTabWidget { background: #2B2B2B; }
QTabWidget::pane { border: 1px solid #3E3E3E; background: #323232; }
QTabBar { background: #2B2B2B; }
QTabBar::scroller { background: #2B2B2B; }
QTabBar QToolButton { background: #2B2B2B; border: none; }
QTabBar::tab { background: #2B2B2B; color: #888888; padding: 6px 14px; border: 1px solid #3E3E3E; border-bottom: none; font-size: 11px; }
QTabBar::tab:selected { background: #323232; color: #E8E8E8; font-weight: bold; border-bottom: 2px solid #E56020; }
QTabBar::tab:hover { background: #383838; color: #D8D8D8; }
QGroupBox { border: 1px solid #3E3E3E; border-radius: 3px; margin-top: 8px; padding-top: 8px; font-weight: bold; font-size: 11px; color: #AAAAAA; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #777777; font-size: 10px; letter-spacing: 1px; }
QPushButton { background-color: #3A3A3A; color: #D8D8D8; border: 1px solid #4A4A4A; border-radius: 2px; padding: 5px 12px; font-size: 11px; }
QPushButton:hover { background-color: #454545; color: #FFFFFF; }
QPushButton:disabled { color: #666666; background: #303030; border-color: #383838; }
QComboBox { border: none; border-bottom: 2px solid #4A4A4A; background: transparent; padding: 4px 2px; font-size: 11px; color: #D8D8D8; }
QComboBox:focus { border-bottom-color: #E56020; }
QComboBox QAbstractItemView { background: #323232; border: 1px solid #4A4A4A; color: #D8D8D8; selection-background-color: #E56020; }
QTableWidget { border: 1px solid #3E3E3E; gridline-color: #3A3A3A; font-size: 11px; background: #323232; alternate-background-color: #2B2B2B; color: #D8D8D8; }
QTableWidget::item { background-color: #323232; color: #D8D8D8; }
QTableWidget::item:alternate { background-color: #2B2B2B; }
QTableWidget::item:selected { background: #E56020; color: white; }
QHeaderView::section { background: #2B2B2B; color: #777777; border: none; border-right: 1px solid #3E3E3E; border-bottom: 1px solid #3E3E3E; padding: 4px 8px; font-size: 10px; font-weight: 600; letter-spacing: 0.5px; }
QScrollBar:vertical { width: 6px; background: transparent; }
QScrollBar::handle:vertical { background: #4A4A4A; border-radius: 3px; min-height: 20px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal { height: 6px; background: transparent; }
QScrollBar::handle:horizontal { background: #4A4A4A; border-radius: 3px; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QLineEdit { border: none; border-bottom: 2px solid #4A4A4A; background: transparent; padding: 4px 2px; font-size: 11px; color: #D8D8D8; }
QLineEdit:focus { border-bottom-color: #E56020; }
QDoubleSpinBox, QSpinBox { border: none; border-bottom: 2px solid #4A4A4A; background: transparent; padding: 4px 2px; font-size: 11px; color: #D8D8D8; }
QSplitter::handle { background: #3E3E3E; }
QDialog { background: #323232; color: #D8D8D8; }
QMessageBox { background: #323232; color: #D8D8D8; }
QScrollArea { background: transparent; border: none; }
"""

GLOBAL_QSS = LIGHT_QSS


class TrainingThread(QThread):
    finished = pyqtSignal(object)
    progress = pyqtSignal(str)

    def __init__(self, data_engine, model_type="RF", max_iter=2000):
        super().__init__()
        self.data_engine = data_engine
        self.model_type = model_type
        self.max_iter = max_iter

    def run(self):
        try:
            self.progress.emit("데이터를 다시 불러오고 전처리하는 중입니다.")
            self.data_engine.load_data()
            self.progress.emit(self.data_engine.format_quality_report())

            X_train, X_test, y_train, _, _, y_raw_test = self.data_engine.preprocess_data()
            if len(X_train) == 0:
                self.finished.emit("전처리 후 학습 가능한 데이터가 없습니다.")
                return

            self.progress.emit(f"{self.model_type} 모델을 초기화하는 중입니다.")
            model_engine = ModelEngine(
                model_type=self.model_type,
                output_dim=y_train.shape[1],
                max_iter=self.max_iter,
            )

            self.progress.emit(f"{self.model_type} 모델을 학습하는 중입니다.")
            model_engine.train(X_train, y_train)

            if not os.path.exists("models"):
                os.makedirs("models")
            model_engine.save("models/material_model.pkl")
            joblib.dump(self.data_engine, "models/data_engine.pkl")

            self.progress.emit("학습 결과를 평가하는 중입니다.")
            mean_scaled, _ = model_engine.predict(X_test)
            y_pred = self.data_engine.inverse_transform_y(mean_scaled)

            from sklearn.metrics import mean_absolute_error, r2_score

            r2 = r2_score(y_raw_test, y_pred, multioutput="raw_values")
            mae = mean_absolute_error(y_raw_test, y_pred, multioutput="raw_values")

            self.finished.emit(
                {
                    "model": model_engine,
                    "metrics": {"r2": r2, "mae": mae},
                    "y_test": y_raw_test,
                    "y_pred": y_pred,
                    "quality_report": self.data_engine.last_quality_report,
                }
            )
        except Exception as exc:
            self.finished.emit(f"학습 중 오류가 발생했습니다: {exc}")


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Materials Discovery Platform")
        self.resize(1400, 900)

        self.data_engine = DataEngine(None)
        self.model_engine = None
        self.model_type = "RF"
        self.preprocessing_ready = False
        self._open_dialogs = []
        self.last_r2_avg = None
        self._dark_mode = False
        self._panel_widgets = []       # background: panel
        self._divider_widgets = []     # background: divider
        self._info_box_widgets = []    # info box (accent left border)
        self._section_lbl_widgets = [] # color: text_label, monospace
        self._muted_bg_widgets = []    # background: summary/muted

        self.init_ui()

    def init_ui(self):
        self.setStyleSheet(GLOBAL_QSS)

        # Menu bar
        mb = self.menuBar()
        mb.setStyleSheet(
            "QMenuBar { background: #252525; color: #BBBBBB; font-size: 11px; padding: 2px 4px; border-bottom: 1px solid #363636; }"
            "QMenuBar::item { background: transparent; padding: 4px 8px; }"
            "QMenuBar::item:selected { background: #363636; color: #FFFFFF; }"
        )
        for name in ["파일", "편집", "보기", "데이터", "분석", "도구", "도움말"]:
            mb.addMenu(name)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Toolbar
        root_layout.addWidget(self._create_toolbar())

        # 3-panel splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(1)

        splitter.addWidget(self._create_left_panel())
        splitter.addWidget(self._create_settings_panel())

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        splitter.addWidget(self.tabs)
        splitter.setSizes([190, 270, 860])

        root_layout.addWidget(splitter, 1)

        # Status bar
        sb = self.statusBar()
        sb.setStyleSheet(
            "QStatusBar { background: #252525; color: white; font-size: 10px; "
            "font-family: 'Consolas', monospace; padding: 0 8px; border-top: 1px solid #363636; }"
            "QStatusBar::item { border: none; }"
        )
        self._sb_status = QLabel("● 준비완료")
        self._sb_samples = QLabel("샘플: —")
        self._sb_missing = QLabel("결측: —")
        self._sb_model = QLabel("모델: 미훈련")
        _lbl_style = "color: #BBBBBB; font-size: 10px; font-family: 'Consolas', monospace; padding: 0 10px;"
        for lbl in [self._sb_samples, self._sb_missing, self._sb_model]:
            lbl.setStyleSheet(_lbl_style)
        self._sb_status.setStyleSheet("color: #4CAF50; font-size: 10px; font-family: 'Consolas', monospace; padding: 0 10px;")
        sb.addWidget(self._sb_status)
        sb.addWidget(self._sb_samples)
        sb.addWidget(self._sb_missing)
        sb.addWidget(self._sb_model)

        self.setup_preprocessing_tab()
        self.setup_feature_selection_tab()
        self.setup_training_tab()
        self.setup_performance_tab()
        self.setup_inference_tab()
        self.setup_workspace_tab()
        self.refresh_workspace_list()

    # ── Toolbar ──────────────────────────────────────────────────────────────

    def _create_toolbar(self):
        bar = QWidget()
        self._toolbar_widget = bar
        bar.setFixedHeight(34)
        bar.setStyleSheet("background: #252525; border-bottom: 1px solid #363636;")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setSpacing(3)

        _tb_style = (
            "QPushButton { background: transparent; color: #BBBBBB; border: 1px solid #363636; "
            "border-radius: 2px; padding: 1px 8px; font-size: 10px; }"
            "QPushButton:hover { background: #363636; color: #FFFFFF; }"
            "QPushButton:disabled { color: #555555; border-color: #2C2C2C; }"
        )

        def tb_btn(text, callback=None, enabled=True):
            b = QPushButton(text)
            b.setFixedHeight(22)
            b.setStyleSheet(_tb_style)
            if callback:
                b.clicked.connect(callback)
            b.setEnabled(enabled)
            return b

        def sep():
            line = QWidget()
            line.setFixedSize(1, 16)
            line.setStyleSheet("background: #363636;")
            return line

        layout.addWidget(tb_btn("열기", self.on_select_file_clicked))
        layout.addWidget(sep())
        layout.addWidget(tb_btn("전처리", self.on_preprocess_clicked))
        layout.addWidget(tb_btn("합금 지표", self.on_generate_features_clicked))
        layout.addWidget(sep())
        layout.addWidget(tb_btn("모델 학습", self.on_train_clicked))
        layout.addWidget(sep())
        layout.addWidget(tb_btn("예측", self.on_predict_clicked))
        layout.addStretch()

        title = QLabel("AI Materials Discovery Platform")
        title.setStyleSheet("color: #555555; font-size: 10px; letter-spacing: 1px;")
        layout.addWidget(title)

        layout.addSpacing(12)
        self._theme_btn = QPushButton("🌙")
        self._theme_btn.setFixedSize(26, 22)
        self._theme_btn.setStyleSheet(
            "QPushButton { background: transparent; color: #BBBBBB; border: 1px solid #363636; "
            "border-radius: 2px; font-size: 12px; padding: 0; }"
            "QPushButton:hover { background: #363636; }"
        )
        self._theme_btn.clicked.connect(self._toggle_theme)
        layout.addWidget(self._theme_btn)
        return bar

    def _toggle_theme(self):
        self._dark_mode = not self._dark_mode
        self._theme_btn.setText("☀️" if self._dark_mode else "🌙")
        self.setStyleSheet(DARK_QSS if self._dark_mode else LIGHT_QSS)
        self._apply_theme_colors()

    def _theme(self):
        d = self._dark_mode
        return {
            "app_bg":      "#2B2B2B" if d else "#F2F2F0",
            "panel_bg":    "#323232" if d else "#FFFFFF",
            "muted_bg":    "#2E2E2E" if d else "#F8F8F7",
            "info_bg":     "#383838" if d else "#F2F2F0",
            "divider":     "#3A3A3A" if d else "#EBEBEA",
            "border":      "#4A4A4A" if d else "#D8D8D6",
            "text_primary":"#E8E8E8" if d else "#1A1A1A",
            "text_sec":    "#BBBBBB" if d else "#444444",
            "text_label":  "#777777" if d else "#8A8A8A",
            "tb_bg":       "#1E1E1E" if d else "#252525",
            "tb_border":   "#2A2A2A" if d else "#363636",
            "accent":      "#E56020",
        }

    def _apply_theme_colors(self):
        c = self._theme()

        # ── Application palette (테이블 교번 행 등 팔레트 의존 위젯) ──
        palette = QPalette()
        if self._dark_mode:
            palette.setColor(QPalette.ColorRole.Window,          QColor("#2B2B2B"))
            palette.setColor(QPalette.ColorRole.WindowText,      QColor("#D8D8D8"))
            palette.setColor(QPalette.ColorRole.Base,            QColor("#323232"))
            palette.setColor(QPalette.ColorRole.AlternateBase,   QColor("#2B2B2B"))
            palette.setColor(QPalette.ColorRole.Text,            QColor("#D8D8D8"))
            palette.setColor(QPalette.ColorRole.Button,          QColor("#3A3A3A"))
            palette.setColor(QPalette.ColorRole.ButtonText,      QColor("#D8D8D8"))
            palette.setColor(QPalette.ColorRole.Highlight,       QColor("#E56020"))
            palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#FFFFFF"))
            palette.setColor(QPalette.ColorRole.ToolTipBase,     QColor("#323232"))
            palette.setColor(QPalette.ColorRole.ToolTipText,     QColor("#D8D8D8"))
        else:
            palette.setColor(QPalette.ColorRole.Window,          QColor("#F2F2F0"))
            palette.setColor(QPalette.ColorRole.WindowText,      QColor("#1A1A1A"))
            palette.setColor(QPalette.ColorRole.Base,            QColor("#FFFFFF"))
            palette.setColor(QPalette.ColorRole.AlternateBase,   QColor("#F8F8F7"))
            palette.setColor(QPalette.ColorRole.Text,            QColor("#1A1A1A"))
            palette.setColor(QPalette.ColorRole.Button,          QColor("#EBEBEA"))
            palette.setColor(QPalette.ColorRole.ButtonText,      QColor("#1A1A1A"))
            palette.setColor(QPalette.ColorRole.Highlight,       QColor("#E56020"))
            palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#FFFFFF"))
        from PyQt6.QtWidgets import QApplication
        QApplication.instance().setPalette(palette)

        # ── Toolbar / MenuBar / StatusBar ──
        self._toolbar_widget.setStyleSheet(
            f"background: {c['tb_bg']}; border-bottom: 1px solid {c['tb_border']};"
        )
        self.menuBar().setStyleSheet(
            f"QMenuBar {{ background: {c['tb_bg']}; color: {c['text_label']}; font-size: 11px; padding: 2px 4px; border-bottom: 1px solid {c['tb_border']}; }}"
            f"QMenuBar::item {{ background: transparent; padding: 4px 8px; }}"
            f"QMenuBar::item:selected {{ background: {c['tb_border']}; color: #FFFFFF; }}"
        )
        self.statusBar().setStyleSheet(
            f"QStatusBar {{ background: {c['tb_bg']}; color: {c['text_sec']}; font-size: 10px; "
            f"font-family: 'Consolas', monospace; padding: 0 8px; border-top: 1px solid {c['tb_border']}; }}"
            "QStatusBar::item { border: none; }"
        )

        # ── Panel backgrounds ──
        for w in self._panel_widgets:
            w.setStyleSheet(f"background: {c['panel_bg']};")

        # ── Dividers ──
        for w in self._divider_widgets:
            w.setStyleSheet(f"background: {c['divider']};")

        # ── Info boxes (accent left border) ──
        for w in self._info_box_widgets:
            w.setStyleSheet(
                f"font-size: 10px; color: {c['text_sec']}; background: {c['info_bg']}; "
                f"padding: 8px; border-left: 2px solid {c['accent']};"
            )

        # ── Section labels (monospace, label color) ──
        for w in self._section_lbl_widgets:
            w.setStyleSheet(
                f"font-size: 10px; color: {c['text_label']}; "
                "font-family: 'Consolas', monospace; letter-spacing: 0.5px;"
            )

        # ── Muted background boxes (summary etc.) ──
        for w in self._muted_bg_widgets:
            w.setStyleSheet(
                f"font-size: 10px; color: {c['text_label']}; padding: 8px; "
                f"background: {c['muted_bg']}; border: 1px solid {c['border']};"
            )

        # ── Settings misc labels ──
        self.file_path_label.setStyleSheet(f"font-size: 11px; color: {c['text_label']};")
        self.status_label.setStyleSheet(f"color: {c['text_label']}; font-size: 10px;")
        self.domain_range_status_label.setStyleSheet(f"color: {c['text_label']}; font-size: 10px;")
        self.reset_preprocess_btn.setStyleSheet(
            f"background: {c['muted_bg']}; color: {c['text_sec']}; border: none; font-size: 11px;"
        )

        # ── Tree labels ──
        for lbl in [self._tree_file_label, self._tree_preprocess_label,
                    self._tree_model_label, self._tree_results_label]:
            cur = lbl.styleSheet()
            lbl.setStyleSheet(cur.replace("#1A1A1A", c["text_primary"])
                                 .replace("#8A8A8A", c["text_label"])
                                 .replace("#E56020", c["accent"]))

        # ── Tree section title labels ──
        for w in self._tree_section_title_lbls:
            w.setStyleSheet(
                f"font-size: 10px; color: {c['text_label']}; font-weight: 600; letter-spacing: 0.5px;"
            )

    # ── Left panel (Project Explorer) ────────────────────────────────────────

    def _create_left_panel(self):
        self._tree_section_title_lbls = []

        panel = QWidget()
        panel.setFixedWidth(190)
        panel.setStyleSheet("background: #FFFFFF;")
        self._panel_widgets.append(panel)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QLabel("프로젝트 탐색기")
        header.setStyleSheet(
            "font-size: 10px; color: #8A8A8A; padding: 8px 12px; letter-spacing: 1px; "
            "border-bottom: 1px solid #EBEBEA;"
        )
        self._panel_widgets.append(header)
        layout.addWidget(header)

        content = QWidget()
        content.setStyleSheet("background: #FFFFFF;")
        self._panel_widgets.append(content)
        cl = QVBoxLayout(content)
        cl.setContentsMargins(12, 8, 12, 8)
        cl.setSpacing(0)

        self._tree_file_label = QLabel("(없음)")
        self._tree_preprocess_label = QLabel("미실행")
        self._tree_model_label = QLabel("미학습")
        self._tree_results_label = QLabel("—")

        def make_section(title_text, value_lbl):
            w = QWidget()
            w.setStyleSheet("background: transparent;")
            wl = QVBoxLayout(w)
            wl.setContentsMargins(0, 6, 0, 6)
            wl.setSpacing(3)
            t = QLabel(title_text)
            t.setStyleSheet("font-size: 10px; color: #8A8A8A; font-weight: 600; letter-spacing: 0.5px;")
            self._tree_section_title_lbls.append(t)
            value_lbl.setStyleSheet("font-size: 11px; color: #1A1A1A; padding-left: 6px;")
            value_lbl.setWordWrap(True)
            wl.addWidget(t)
            wl.addWidget(value_lbl)
            div = QWidget()
            div.setFixedHeight(1)
            div.setStyleSheet("background: #EBEBEA;")
            self._divider_widgets.append(div)
            wl.addWidget(div)
            return w

        cl.addWidget(make_section("데이터 파일", self._tree_file_label))
        cl.addWidget(make_section("전처리", self._tree_preprocess_label))
        cl.addWidget(make_section("모델", self._tree_model_label))
        cl.addWidget(make_section("결과", self._tree_results_label))
        cl.addStretch()

        layout.addWidget(content, 1)
        return panel

    def _update_project_tree(self):
        if self.data_engine.file_path:
            self._tree_file_label.setText(os.path.basename(self.data_engine.file_path))
            self._tree_file_label.setStyleSheet("font-size: 11px; color: #1A1A1A; padding-left: 6px;")
        else:
            self._tree_file_label.setText("(없음)")
            self._tree_file_label.setStyleSheet("font-size: 11px; color: #8A8A8A; padding-left: 6px;")

        if self.preprocessing_ready:
            n = self.data_engine.last_quality_report.get("rows_after", "?")
            self._tree_preprocess_label.setText(f"완료 ({n}행)")
            self._tree_preprocess_label.setStyleSheet("font-size: 11px; color: #E56020; padding-left: 6px; font-weight: bold;")
        else:
            self._tree_preprocess_label.setText("미실행")
            self._tree_preprocess_label.setStyleSheet("font-size: 11px; color: #8A8A8A; padding-left: 6px;")

        if self.model_engine:
            nm = {"RF": "Random Forest", "GBM": "Gradient Boosting", "MLP": "Neural Network", "TFP": "TFP"}
            self._tree_model_label.setText(nm.get(self.model_type, self.model_type))
            self._tree_model_label.setStyleSheet("font-size: 11px; color: #E56020; padding-left: 6px; font-weight: bold;")
        else:
            self._tree_model_label.setText("미학습")
            self._tree_model_label.setStyleSheet("font-size: 11px; color: #8A8A8A; padding-left: 6px;")

        if self.last_r2_avg is not None:
            self._tree_results_label.setText(f"R² {self.last_r2_avg * 100:.1f}%")
            self._tree_results_label.setStyleSheet("font-size: 11px; color: #E56020; padding-left: 6px; font-weight: bold;")
        else:
            self._tree_results_label.setText("—")
            self._tree_results_label.setStyleSheet("font-size: 11px; color: #8A8A8A; padding-left: 6px;")

        if self.data_engine.last_quality_report:
            s = self.data_engine.last_quality_report.get("rows_after", 0)
            self._sb_samples.setText(f"샘플: {s}")
        if self.model_engine:
            nm = {"RF": "Random Forest", "GBM": "Gradient Boosting", "MLP": "Neural Network", "TFP": "TFP"}
            self._sb_model.setText(f"모델: {nm.get(self.model_type, self.model_type)}")

    # ── Settings panel (Middle) ───────────────────────────────────────────────

    def _create_settings_panel(self):
        panel = QWidget()
        panel.setStyleSheet("background: #FFFFFF;")
        self._panel_widgets.append(panel)
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        hdr_lbl = QLabel("설정")
        hdr_lbl.setStyleSheet(
            "font-size: 10px; color: #8A8A8A; padding: 8px 16px; letter-spacing: 1px; "
            "border-bottom: 1px solid #EBEBEA;"
        )
        self._panel_widgets.append(hdr_lbl)
        help_btn = QPushButton("?")
        help_btn.setFixedSize(20, 20)
        help_btn.setStyleSheet(
            "QPushButton { background: transparent; color: #8A8A8A; border: 1px solid #D8D8D6; "
            "border-radius: 10px; font-size: 10px; padding: 0; margin: 4px 8px 4px 0; }"
            "QPushButton:hover { background: #EBEBEA; color: #1A1A1A; }"
        )
        help_btn.clicked.connect(self.show_quality_help)
        hdr_row = QHBoxLayout()
        hdr_row.addWidget(hdr_lbl)
        hdr_row.addStretch()
        hdr_row.addWidget(help_btn)
        outer.addLayout(hdr_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setStyleSheet("background: transparent; border: none;")

        content = QWidget()
        content.setStyleSheet("background: #FFFFFF;")
        self._panel_widgets.append(content)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(14)

        def s_label(text):
            lbl = self._s_label(text)
            self._section_lbl_widgets.append(lbl)
            return lbl

        def s_divider():
            d = self._s_divider()
            self._divider_widgets.append(d)
            return d

        # 01 데이터 소스
        layout.addWidget(s_label("01 — 데이터 소스"))
        self.file_path_label = QLabel("파일: 선택되지 않음")
        self.file_path_label.setWordWrap(True)
        self.file_path_label.setStyleSheet("font-size: 11px; color: #8A8A8A;")
        layout.addWidget(self.file_path_label)
        self.select_file_btn = QPushButton("파일 열기  (.xls / .xlsx)")
        self.select_file_btn.setFixedHeight(34)
        self.select_file_btn.setStyleSheet(
            "background: #E56020; color: white; border: none; font-size: 11px; font-weight: 600;"
        )
        self.select_file_btn.clicked.connect(self.on_select_file_clicked)
        layout.addWidget(self.select_file_btn)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #8A8A8A; font-size: 10px;")
        layout.addWidget(self.status_label)

        layout.addWidget(s_divider())

        # 02 도메인 검증
        layout.addWidget(s_label("02 — 도메인 검증"))
        self.domain_rule_label = QLabel(
            "오스테나이트 조성 기준과 고온 특성 기준 두 부류로 범위를 확인합니다."
        )
        self.domain_rule_label.setWordWrap(True)
        self.domain_rule_label.setStyleSheet(
            "font-size: 10px; color: #444444; background: #F2F2F0; padding: 8px; border-left: 2px solid #E56020;"
        )
        self._info_box_widgets.append(self.domain_rule_label)
        layout.addWidget(self.domain_rule_label)
        domain_row = QHBoxLayout()
        self.austenite_domain_btn = QPushButton("오스테나이트")
        self.austenite_domain_btn.clicked.connect(self.show_austenite_domain_dialog)
        self.high_temp_domain_btn = QPushButton("고온 특성")
        self.high_temp_domain_btn.clicked.connect(self.show_high_temp_domain_dialog)
        domain_row.addWidget(self.austenite_domain_btn)
        domain_row.addWidget(self.high_temp_domain_btn)
        layout.addLayout(domain_row)
        self.domain_range_status_label = QLabel("")
        self.domain_range_status_label.setWordWrap(True)
        self.domain_range_status_label.setStyleSheet("color: #8A8A8A; font-size: 10px;")
        layout.addWidget(self.domain_range_status_label)
        self.refresh_domain_range_status()

        layout.addWidget(s_divider())

        # 03 데이터 품질
        layout.addWidget(s_label("03 — 데이터 품질"))
        form = QFormLayout()
        form.setSpacing(10)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFormAlignment(Qt.AlignmentFlag.AlignLeft)
        self.missing_combo = QComboBox()
        self.missing_combo.addItems(["평균값으로 채우기", "중앙값으로 채우기", "주변 값으로 예측(KNN)", "해당 행 제거"])
        form.addRow("결측값:", self.missing_combo)
        self.outlier_combo = QComboBox()
        self.outlier_combo.addItems(["감지 범위로 보정", "이상치 행 제거", "표시만 하고 유지"])
        form.addRow("이상치:", self.outlier_combo)
        self.invalid_type_combo = QComboBox()
        self.invalid_type_combo.addItems(["잘못된 값을 NaN으로 변환", "잘못된 값이 있는 행 제거"])
        form.addRow("형식 검증:", self.invalid_type_combo)
        self.iqr_spin = QDoubleSpinBox()
        self.iqr_spin.setRange(0.5, 5.0)
        self.iqr_spin.setSingleStep(0.1)
        self.iqr_spin.setValue(1.5)
        form.addRow("IQR 민감도:", self.iqr_spin)
        layout.addLayout(form)

        layout.addWidget(s_divider())

        # 04 합금 지표
        layout.addWidget(s_label("04 — 합금 지표"))
        self.feature_engineering_check = QCheckBox("합금 지표 생성 사용")
        self.feature_engineering_check.setChecked(True)
        self.feature_engineering_check.setVisible(False)
        layout.addWidget(self.feature_engineering_check)
        self.feature_engineering_label = QLabel("Cr/Ni, C+N, Ni_eq, Cr_eq를 자동 생성합니다.")
        self.feature_engineering_label.setWordWrap(True)
        self.feature_engineering_label.setStyleSheet(
            "font-size: 10px; color: #444444; background: #F2F2F0; padding: 8px; border-left: 2px solid #E56020;"
        )
        self._info_box_widgets.append(self.feature_engineering_label)
        layout.addWidget(self.feature_engineering_label)

        self.quality_summary_label = QLabel("전처리 결과 요약이 아직 없습니다.")
        self.quality_summary_label.setWordWrap(True)
        self.quality_summary_label.setStyleSheet(
            "font-size: 10px; color: #8A8A8A; padding: 8px; background: #F8F8F7; border: 1px solid #D8D8D6;"
        )
        self._muted_bg_widgets.append(self.quality_summary_label)
        layout.addWidget(self.quality_summary_label)

        layout.addWidget(s_divider())

        # Action buttons
        self.preprocess_btn = QPushButton("전처리 실행")
        self.preprocess_btn.setFixedHeight(38)
        self.preprocess_btn.setStyleSheet(
            "background: #E56020; color: white; border: none; font-size: 12px; font-weight: 700; letter-spacing: 1px;"
        )
        self.preprocess_btn.clicked.connect(self.on_preprocess_clicked)
        layout.addWidget(self.preprocess_btn)

        self.generate_features_btn = QPushButton("합금 지표 생성")
        self.generate_features_btn.setFixedHeight(34)
        self.generate_features_btn.setEnabled(False)
        self.generate_features_btn.clicked.connect(self.on_generate_features_clicked)
        layout.addWidget(self.generate_features_btn)

        self.reset_preprocess_btn = QPushButton("전처리 초기화")
        self.reset_preprocess_btn.setFixedHeight(30)
        self.reset_preprocess_btn.setStyleSheet(
            "background: #EBEBEA; color: #444444; border: none; font-size: 11px;"
        )
        self.reset_preprocess_btn.clicked.connect(self.on_reset_preprocessing_clicked)
        layout.addWidget(self.reset_preprocess_btn)

        self.go_to_training_btn = QPushButton("→ 학습 컬럼 선택으로")
        self.go_to_training_btn.setFixedHeight(30)
        self.go_to_training_btn.setEnabled(False)
        self.go_to_training_btn.clicked.connect(lambda: self.tabs.setCurrentIndex(1))
        layout.addWidget(self.go_to_training_btn)

        layout.addStretch()

        # Signal connections
        self.missing_combo.currentIndexChanged.connect(self.mark_preprocessing_dirty)
        self.outlier_combo.currentIndexChanged.connect(self.mark_preprocessing_dirty)
        self.invalid_type_combo.currentIndexChanged.connect(self.mark_preprocessing_dirty)
        self.iqr_spin.valueChanged.connect(self.mark_preprocessing_dirty)
        self.feature_engineering_check.stateChanged.connect(self.mark_preprocessing_dirty)

        scroll.setWidget(content)
        outer.addWidget(scroll, 1)
        return panel

    def _s_label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet("font-size: 10px; color: #8A8A8A; font-family: 'Consolas', monospace; letter-spacing: 0.5px;")
        return lbl

    def _s_divider(self):
        div = QWidget()
        div.setFixedHeight(1)
        div.setStyleSheet("background: #EBEBEA;")
        return div

    def setup_preprocessing_tab(self):
        tab = QWidget()
        tab.setStyleSheet("background: #FFFFFF;")
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.processed_result_tabs = QTabWidget()
        self.processed_result_tabs.setDocumentMode(True)

        self.processed_preview_table = QTableWidget()
        self.processed_preview_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.processed_preview_table.setAlternatingRowColors(True)
        self.processed_preview_table.verticalHeader().setVisible(False)
        self.processed_preview_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.processed_result_tabs.addTab(self.processed_preview_table, "데이터 전처리 결과")

        self.engineered_preview_table = QTableWidget()
        self.engineered_preview_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.engineered_preview_table.setAlternatingRowColors(True)
        self.engineered_preview_table.verticalHeader().setVisible(False)
        self.engineered_preview_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.processed_result_tabs.addTab(self.engineered_preview_table, "합금 지표 생성 결과")

        layout.addWidget(self.processed_result_tabs, 1)

        self.processed_preview_info_label = QLabel(
            "전처리를 실행하면 처리 완료된 전체 데이터를 아래 표에서 확인할 수 있습니다."
        )
        self.processed_preview_info_label.setWordWrap(True)
        self.processed_preview_info_label.setStyleSheet(
            "font-size: 10px; color: #8A8A8A; padding: 4px 10px; background: #F8F8F7; border-top: 1px solid #EBEBEA;"
        )
        layout.addWidget(self.processed_preview_info_label)

        self.tabs.addTab(tab, "데이터 미리보기")

    def setup_feature_selection_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        intro_label = QLabel(
            "모델 학습에 사용할 컬럼을 선택합니다. 체크한 컬럼만 학습과 예측에 사용됩니다."
        )
        intro_label.setWordWrap(True)
        intro_label.setStyleSheet(
            "background-color: #eef6ff; padding: 12px; border-radius: 8px; color: #355c7d;"
        )
        layout.addWidget(intro_label)

        self.feature_selection_status_label = QLabel(
            "먼저 전처리를 실행한 뒤, 이 탭에서 학습 컬럼을 선택해 주세요."
        )
        self.feature_selection_status_label.setWordWrap(True)
        self.feature_selection_status_label.setStyleSheet(
            "background-color: #fff7d6; padding: 10px; border-radius: 8px; color: #7a5d00;"
        )
        layout.addWidget(self.feature_selection_status_label)

        button_row = QHBoxLayout()
        self.select_all_features_btn = QPushButton("전체 선택")
        self.select_all_features_btn.setEnabled(False)
        self.select_all_features_btn.clicked.connect(self.select_all_feature_columns)
        button_row.addWidget(self.select_all_features_btn)

        self.clear_features_btn = QPushButton("전체 해제")
        self.clear_features_btn.setEnabled(False)
        self.clear_features_btn.clicked.connect(self.clear_all_feature_columns)
        button_row.addWidget(self.clear_features_btn)

        button_row.addStretch()

        self.go_to_model_training_btn = QPushButton("모델 학습 탭으로 이동")
        self.go_to_model_training_btn.setEnabled(False)
        self.go_to_model_training_btn.clicked.connect(lambda: self.tabs.setCurrentIndex(2))
        button_row.addWidget(self.go_to_model_training_btn)
        layout.addLayout(button_row)

        self.feature_selection_table = QTableWidget()
        self.feature_selection_table.setColumnCount(3)
        self.feature_selection_table.setHorizontalHeaderLabels(["사용", "컬럼", "구분"])
        self.feature_selection_table.verticalHeader().setVisible(False)
        self.feature_selection_table.setAlternatingRowColors(True)
        self.feature_selection_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.feature_selection_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.feature_selection_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.feature_selection_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.feature_selection_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.feature_selection_table.itemChanged.connect(self.on_feature_selection_item_changed)
        layout.addWidget(self.feature_selection_table)

        self.tabs.addTab(tab, "학습 컬럼 선택")

    def setup_training_tab(self):
        tab = QWidget()
        outer_layout = QVBoxLayout(tab)

        top_row = QHBoxLayout()
        top_row.addStretch()
        model_help_btn = QPushButton("모델 학습 도움말")
        model_help_btn.setFixedWidth(140)
        model_help_btn.clicked.connect(self.show_model_training_help)
        top_row.addWidget(model_help_btn)
        outer_layout.addLayout(top_row)

        content_layout = QHBoxLayout()
        left_panel = QVBoxLayout()

        info_group = QGroupBox("2. 모델 학습")
        info_layout = QVBoxLayout(info_group)
        info_layout.setSpacing(10)

        self.training_data_status_label = QLabel("")
        self.training_data_status_label.setWordWrap(True)
        self.training_data_status_label.setStyleSheet(
            "background-color: #fff7d6; padding: 10px; border-radius: 8px; color: #7a5d00;"
        )
        info_layout.addWidget(self.training_data_status_label)

        model_selection_group = QGroupBox("AI 모델 학습 설정")
        model_form = QFormLayout(model_selection_group)

        self.model_combo = QComboBox()
        self.model_combo.addItems(
            ["Random Forest", "Gradient Boosting", "Neural Network", "TFP"]
        )
        model_form.addRow("학습 모델:", self.model_combo)

        self.iter_spin = QSpinBox()
        self.iter_spin.setRange(100, 10000)
        self.iter_spin.setValue(2000)
        self.iter_spin.setSingleStep(500)
        model_form.addRow("최대 반복 횟수:", self.iter_spin)
        self.training_input_combo = QComboBox()
        self.training_input_combo.addItems(
            ["데이터 정제 + 합금 지표", "데이터 정제만", "합금 지표만"]
        )
        model_form.addRow("학습 데이터 선택:", self.training_input_combo)
        info_layout.addWidget(model_selection_group)

        help_label = QLabel(
            "반복 횟수를 크게 늘린다고 항상 성능이 좋아지지는 않습니다.\n기본값으로 먼저 학습한 뒤 필요할 때만 조정하는 것을 권장합니다."
        )
        help_label.setWordWrap(True)
        help_label.setStyleSheet(
            "font-family: 'Malgun Gothic'; font-size: 10px; color: #3498db; font-style: italic; line-height: 1.3; background-color: #eef6ff; padding: 8px; border-radius: 6px;"
        )
        info_layout.addWidget(help_label)

        self.training_status_label = QLabel("")
        self.training_status_label.setWordWrap(True)
        self.training_status_label.setStyleSheet("color: #34495e; padding: 2px 0 6px 0;")
        info_layout.addWidget(self.training_status_label)

        self.train_btn = QPushButton("모델 학습 시작")
        self.train_btn.setFixedHeight(45)
        self.train_btn.setEnabled(False)
        self.train_btn.setStyleSheet(
            "background-color: #3498db; color: white; font-weight: bold; font-size: 13px; border-radius: 8px;"
        )
        self.train_btn.clicked.connect(self.on_train_clicked)
        info_layout.addWidget(self.train_btn)

        self.metrics_label = QLabel("<b>모델 성능 요약:</b><br>- 예측 정확도: N/A<br>- 평균 오차: N/A")
        self.metrics_label.setWordWrap(True)
        self.metrics_label.setStyleSheet(
            "background-color: #f8f9fa; padding: 12px; border-radius: 8px; border: 1px solid #e2e8f0;"
        )
        info_layout.addWidget(self.metrics_label)

        left_panel.addWidget(info_group)
        left_panel.addStretch()

        right_panel = QVBoxLayout()
        self.canvas = MplCanvas(self, width=6.6, height=5.2, dpi=100)
        right_panel.addWidget(self.canvas)
        self.render_training_placeholder()

        content_layout.addLayout(left_panel, 1)
        content_layout.addLayout(right_panel, 2)
        outer_layout.addLayout(content_layout)
        self.tabs.addTab(tab, "모델 학습")

    def setup_performance_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        header = QLabel("상세 성능 분석 (Predicted vs Actual)")
        header.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(header)

        self.perf_canvas = FigureCanvas(Figure(figsize=(10, 8)))
        layout.addWidget(self.perf_canvas)

        desc = QLabel("* 학습이 끝나면 실제값과 예측값 비교 그래프가 여기에 표시됩니다.")
        desc.setStyleSheet("color: #7f8c8d; font-style: italic;")
        layout.addWidget(desc)

        self.tabs.addTab(tab, "상세 성능 분석")
        self.render_performance_placeholder()

    def setup_inference_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        left_widget = QWidget()
        left_panel = QVBoxLayout(left_widget)
        scroll.setWidget(left_widget)

        self.inputs = {}
        self.active_model_info = QLabel("사용 중인 모델: 학습된 모델 없음")
        self.active_model_info.setStyleSheet(
            "background-color: #fcf3cf; padding: 10px; border: 1px solid #f39c12; border-radius: 5px; font-weight: bold; margin-bottom: 10px;"
        )
        left_panel.addWidget(self.active_model_info)

        comp_group = QGroupBox("Chemical Composition (wt%)")
        comp_layout = QFormLayout(comp_group)
        comp_list = [
            "Cr", "Ni", "Mo", "Mn", "Si", "Nb", "Ti", "Zr", "Ta", "V", "W", "Cu", "N", "C", "B", "P", "S", "Co", "Al", "Sn", "Pb",
        ]
        default_map = {"Cr": "18.0", "Ni": "8.0", "Mn": "2.0", "Si": "1.0", "C": "0.08"}
        for col in comp_list:
            line_edit = QLineEdit()
            line_edit.setText(default_map.get(col, "0.0"))
            comp_layout.addRow(QLabel(col), line_edit)
            self.inputs[col] = line_edit
        left_panel.addWidget(comp_group)

        proc_group = QGroupBox("Process & Structure")
        proc_layout = QFormLayout(proc_group)
        proc_defaults = {
            "Solution_treatment_temperature": "1050",
            "Solution_treatment_time(s)": "3600",
            "Water_Quenched_after_s.t.": "1",
            "Air_Quenched_after_s.t.": "0",
            "Grains mm-2": "500",
            "Type of melting": "2",
            "Size of ingot": "50",
            "Product form": "3",
            "Temperature (K)": "300",
        }
        for col, value in proc_defaults.items():
            line_edit = QLineEdit()
            line_edit.setText(value)
            proc_layout.addRow(QLabel(col), line_edit)
            self.inputs[col] = line_edit
        left_panel.addWidget(proc_group)

        predict_btn = QPushButton("Predict Properties")
        predict_btn.setFixedHeight(45)
        predict_btn.setStyleSheet("background-color: #2ecc71; color: white; font-weight: bold; margin-top: 10px;")
        predict_btn.clicked.connect(self.on_predict_clicked)
        left_panel.addWidget(predict_btn)
        left_panel.addStretch()

        right_panel = QVBoxLayout()
        result_group = QGroupBox("Predictions")
        result_layout = QVBoxLayout(result_group)

        self.result_display = QLabel("Enter parameters and click Predict.")
        self.result_display.setFont(QFont("Arial", 12))
        result_layout.addWidget(self.result_display)

        self.prediction_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        result_layout.addWidget(self.prediction_canvas)
        right_panel.addWidget(result_group)

        layout.addWidget(scroll, 1)
        layout.addLayout(right_panel, 1)
        self.tabs.addTab(tab, "물성 예측")

    def setup_workspace_tab(self):
        tab = QWidget()
        tab.setStyleSheet("background: #FFFFFF;")
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        # ── 저장/불러오기 컨트롤 (이전 상단 바에서 이동) ──────────────
        ws_save_row = QHBoxLayout()
        ws_save_row.setSpacing(6)
        name_lbl = QLabel("이름:")
        name_lbl.setStyleSheet("font-size: 11px; color: #8A8A8A;")
        self.ws_name_input = QLineEdit()
        self.ws_name_input.setPlaceholderText("분석 저장 이름 입력 (예: 실험A)")
        self.ws_name_input.setFixedWidth(200)
        save_ws_btn = QPushButton("저장")
        save_ws_btn.setStyleSheet(
            "background: #E56020; color: white; border: none; font-weight: bold; padding: 5px 14px;"
        )
        save_ws_btn.clicked.connect(self.save_workspace)
        self.ws_combo = QComboBox()
        self.ws_combo.setFixedWidth(200)
        load_ws_btn = QPushButton("불러오기")
        load_ws_btn.clicked.connect(self.load_workspace)
        delete_ws_btn = QPushButton("삭제")
        delete_ws_btn.setStyleSheet(
            "background: #D32F2F; color: white; border: none; font-weight: bold; padding: 5px 10px;"
        )
        delete_ws_btn.clicked.connect(self.delete_workspace)
        ws_save_row.addWidget(name_lbl)
        ws_save_row.addWidget(self.ws_name_input)
        ws_save_row.addWidget(save_ws_btn)
        ws_save_row.addSpacing(16)
        ws_save_row.addWidget(self.ws_combo)
        ws_save_row.addWidget(load_ws_btn)
        ws_save_row.addWidget(delete_ws_btn)
        ws_save_row.addStretch()
        layout.addLayout(ws_save_row)

        # ── 목록 헤더 ──────────────────────────────────────────────────
        header_row = QHBoxLayout()
        title = QLabel("분석 저장 목록")
        title.setStyleSheet("font-size: 13px; font-weight: bold; color: #1A1A1A;")
        header_row.addWidget(title)
        header_row.addStretch()
        compare_btn = QPushButton("선택 비교 (최대 3개)")
        compare_btn.setFixedWidth(150)
        compare_btn.setStyleSheet("background: #8e44ad; color: white; border: none; font-weight: bold; padding: 5px;")
        compare_btn.clicked.connect(self._on_compare_clicked)
        header_row.addWidget(compare_btn)
        refresh_btn = QPushButton("새로고침")
        refresh_btn.setFixedWidth(90)
        refresh_btn.clicked.connect(self.refresh_workspace_table)
        header_row.addWidget(refresh_btn)
        layout.addLayout(header_row)

        # 상단 테이블 + 하단 상세 패널을 QSplitter로 분리
        splitter = QSplitter(Qt.Orientation.Vertical)

        # ── 상단: 목록 테이블 ──
        self.ws_table = QTableWidget()
        self.ws_table.setColumnCount(5)
        self.ws_table.setHorizontalHeaderLabels(["이름", "모델", "데이터 파일", "저장 날짜", "R2 평균"])
        self.ws_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.ws_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.ws_table.setSelectionMode(QTableWidget.SelectionMode.MultiSelection)
        self.ws_table.setAlternatingRowColors(True)
        self.ws_table.verticalHeader().setVisible(False)
        self.ws_table.horizontalHeader().setStretchLastSection(True)
        self.ws_table.setColumnWidth(0, 160)
        self.ws_table.setColumnWidth(1, 130)
        self.ws_table.setColumnWidth(2, 200)
        self.ws_table.setColumnWidth(3, 160)
        self.ws_table.cellClicked.connect(self._on_ws_table_clicked)
        self.ws_table.cellDoubleClicked.connect(self._on_ws_table_double_clicked)
        splitter.addWidget(self.ws_table)

        # ── 하단: 상세 정보 패널 ──
        detail_widget = QWidget()
        detail_widget.setStyleSheet("background-color: #f6f8fa; border-top: 1px solid #dde1e6;")
        detail_layout = QHBoxLayout(detail_widget)
        detail_layout.setContentsMargins(12, 10, 12, 10)

        # 그래프 썸네일 영역 (학습 + 상세 성능)
        thumb_col = QVBoxLayout()
        thumb_col.setSpacing(4)

        # 학습 그래프 썸네일
        train_lbl = QLabel("학습 그래프")
        train_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        train_lbl.setStyleSheet("font-size: 10px; color: #7f8c8d;")
        thumb_col.addWidget(train_lbl)
        self.ws_detail_thumb = QLabel()
        self.ws_detail_thumb.setFixedSize(200, 120)
        self.ws_detail_thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ws_detail_thumb.setStyleSheet("border: 1px solid #dde1e6; background: white;")
        self.ws_detail_thumb.setText("없음")
        self.ws_detail_thumb.setCursor(Qt.CursorShape.PointingHandCursor)
        self.ws_detail_thumb.mousePressEvent = self._on_thumb_clicked
        self._ws_thumb_full_path = None
        thumb_col.addWidget(self.ws_detail_thumb)

        # 상세 성능 그래프 썸네일
        perf_lbl = QLabel("상세 성능")
        perf_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        perf_lbl.setStyleSheet("font-size: 10px; color: #7f8c8d;")
        thumb_col.addWidget(perf_lbl)
        self.ws_detail_perf_thumb = QLabel()
        self.ws_detail_perf_thumb.setFixedSize(200, 120)
        self.ws_detail_perf_thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ws_detail_perf_thumb.setStyleSheet("border: 1px solid #dde1e6; background: white;")
        self.ws_detail_perf_thumb.setText("없음")
        self.ws_detail_perf_thumb.setCursor(Qt.CursorShape.PointingHandCursor)
        self.ws_detail_perf_thumb.mousePressEvent = self._on_perf_thumb_clicked
        self._ws_perf_thumb_full_path = None
        thumb_col.addWidget(self.ws_detail_perf_thumb)

        detail_layout.addLayout(thumb_col)

        # 상세 텍스트
        self.ws_detail_info = QLabel()
        self.ws_detail_info.setWordWrap(True)
        self.ws_detail_info.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.ws_detail_info.setStyleSheet("font-size: 12px; color: #2c3e50; padding-left: 12px;")
        self.ws_detail_info.setText("저장된 분석을 클릭하면 상세 정보가 표시됩니다.")
        detail_layout.addWidget(self.ws_detail_info, 1)

        # 불러오기 버튼
        load_btn = QPushButton("이 분석 불러오기")
        load_btn.setFixedSize(130, 40)
        load_btn.setStyleSheet("background-color: #16a085; color: white; font-weight: bold;")
        load_btn.clicked.connect(self._load_selected_ws)
        detail_layout.addWidget(load_btn, 0, Qt.AlignmentFlag.AlignBottom)

        splitter.addWidget(detail_widget)
        splitter.setSizes([400, 180])
        layout.addWidget(splitter)

        hint = QLabel("※ 행 클릭 → 상세 보기  |  더블클릭 → 바로 불러오기  |  Ctrl+클릭으로 여러 행 선택 후 [선택 비교] 클릭")
        hint.setStyleSheet("color: #7f8c8d; font-size: 11px; margin-top: 4px;")
        layout.addWidget(hint)

        self.tabs.addTab(tab, "워크스페이스")
        self.refresh_workspace_table()

    def refresh_domain_range_status(self):
        custom_count = len(getattr(self.data_engine, "custom_ranges", {}))
        total_count = len(self.data_engine.get_domain_ranges())
        if custom_count:
            self.domain_range_status_label.setText(
                f"현재 사용자 지정 기준 {custom_count}개가 적용되어 있습니다. 검증 대상 컬럼은 총 {total_count}개입니다."
            )
        else:
            self.domain_range_status_label.setText(
                f"현재 기본 도메인 기준으로 검증합니다. 검증 대상 컬럼은 총 {total_count}개입니다."
            )

    def mark_preprocessing_dirty(self, *_args):
        self.preprocessing_ready = False
        self.train_btn.setEnabled(False)
        self.go_to_training_btn.setEnabled(False)
        if hasattr(self, "go_to_model_training_btn"):
            self.go_to_model_training_btn.setEnabled(False)
        if hasattr(self, "generate_features_btn"):
            self.generate_features_btn.setEnabled(False)
        if self.data_engine.df is not None and not self.data_engine.df.empty:
            self.training_data_status_label.setText("현재 표시 중인 결과는 이전 설정 기준입니다. 새 설정으로 다시 전처리를 실행해 주세요.")
            self.training_status_label.setText("상태: 설정이 변경되어 학습이 잠시 비활성화되었습니다. 전처리를 다시 실행해 주세요.")
            self.quality_summary_label.setText("전처리 설정이 변경되었습니다. 현재 표는 이전 설정 기준 결과입니다.")
            self.processed_preview_info_label.setText("현재 표는 이전 전처리 결과입니다. 새 설정을 반영하려면 전처리를 다시 실행해 주세요.")
            return
        self.training_data_status_label.setText("")
        self.training_status_label.setText("")
        self.quality_summary_label.setText("전처리 설정이 변경되었습니다. 전처리를 다시 실행해 주세요.")
        self.processed_preview_info_label.setText("설정이 변경되었고 아직 전처리 결과가 없습니다. 전처리를 다시 실행해 주세요.")

        if hasattr(self, "feature_selection_status_label"):
            self.feature_selection_status_label.setText(
                "먼저 전처리를 실행한 뒤, 이 탭에서 학습 컬럼을 선택해 주세요."
            )

    def apply_quality_settings_from_ui(self):
        missing_map = {0: "mean", 1: "median", 2: "knn", 3: "drop"}
        outlier_map = {0: "clip", 1: "remove", 2: "flag"}
        invalid_type_map = {0: "coerce", 1: "drop"}
        input_feature_mode_map = {
            0: "combined",
            1: "clean_only",
            2: "engineered_only",
        }
        self.data_engine.configure_quality_rules(
            missing_strategy=missing_map.get(self.missing_combo.currentIndex(), "mean"),
            outlier_strategy=outlier_map.get(self.outlier_combo.currentIndex(), "clip"),
            invalid_type_strategy=invalid_type_map.get(self.invalid_type_combo.currentIndex(), "coerce"),
            iqr_factor=self.iqr_spin.value(),
            input_feature_mode=input_feature_mode_map.get(self.training_input_combo.currentIndex(), "combined"),
        )

    def update_quality_summary_from_report(self, report):
        if not report:
            self.quality_summary_label.setText("전처리 결과 요약이 아직 없습니다.")
            self.domain_rule_label.setText("도메인 기준 검증 결과가 아직 없습니다.")
            return

        self.quality_summary_label.setText(
            "데이터 품질 처리 결과: "
            f"행 {report.get('rows_before', 0)} -> {report.get('rows_after', 0)}, "
            f"형식 오류 {report.get('invalid_type_cells', 0)}개, "
            f"누락값 {report.get('missing_cells_before', 0)} -> {report.get('missing_cells_after', 0)}, "
            f"이상치 감지 {report.get('outlier_cells', 0)}개, "
            f"합금 지표 생성 {len(report.get('engineered_features_added', []))}개"
        )
        self.domain_rule_label.setText(
            "도메인 기준 검증 결과: "
            f"{report.get('domain_range_cells', 0)}개의 값이 설정 범위를 벗어났습니다."
        )

    def populate_processed_preview(self, df):
        base_df = self.data_engine.get_preprocessed_display_df()
        engineered_df = self.data_engine.get_engineered_display_df()
        self.processed_preview_table.clear()
        self.processed_preview_table.setRowCount(len(base_df))
        self.processed_preview_table.setColumnCount(len(base_df.columns))
        self.processed_preview_table.setHorizontalHeaderLabels([str(col) for col in base_df.columns])

        for row_index, (_, row) in enumerate(base_df.iterrows()):
            for col_index, value in enumerate(row):
                if pd.isna(value):
                    text = ""
                elif isinstance(value, (int, float, np.integer, np.floating)):
                    text = f"{float(value):.4g}"
                else:
                    text = str(value)
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.processed_preview_table.setItem(row_index, col_index, item)

        self.processed_preview_table.resizeColumnsToContents()
        self.engineered_preview_table.clear()
        self.engineered_preview_table.setRowCount(len(engineered_df))
        self.engineered_preview_table.setColumnCount(len(engineered_df.columns))
        self.engineered_preview_table.setHorizontalHeaderLabels([str(col) for col in engineered_df.columns])

        for row_index, (_, row) in enumerate(engineered_df.iterrows()):
            for col_index, value in enumerate(row):
                if pd.isna(value):
                    text = ""
                elif isinstance(value, (int, float, np.integer, np.floating)):
                    text = f"{float(value):.4g}"
                else:
                    text = str(value)
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.engineered_preview_table.setItem(row_index, col_index, item)

        self.engineered_preview_table.resizeColumnsToContents()
        self.processed_preview_info_label.setText(
            f"전처리 완료 데이터 전체 결과: 총 {len(df)}행 표시"
        )

    def render_training_placeholder(self):
        self.canvas.axes.clear()
        self.canvas.axes.axis("off")
        self.canvas.axes.text(0.5, 0.58, "2번 탭에서 모델 학습 결과가 여기에 표시됩니다.", ha="center", va="center", fontsize=14, color="#5d6d7e", transform=self.canvas.axes.transAxes)
        self.canvas.axes.text(0.5, 0.42, "먼저 1번 탭에서 전처리를 실행해 주세요.", ha="center", va="center", fontsize=11, color="#85929e", transform=self.canvas.axes.transAxes)
        self.canvas.draw()

    def render_performance_placeholder(self):
        self.perf_canvas.figure.clear()
        ax = self.perf_canvas.figure.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.56, "모델 학습이 끝나면 상세 성능 분석 그래프가 여기에 표시됩니다.", ha="center", va="center", fontsize=14, color="#5d6d7e", transform=ax.transAxes)
        ax.text(0.5, 0.40, "실제값과 예측값이 얼마나 비슷한지 특성별로 확인할 수 있습니다.", ha="center", va="center", fontsize=11, color="#85929e", transform=ax.transAxes)
        self.perf_canvas.figure.tight_layout()
        self.perf_canvas.draw()

    def populate_feature_selection_table(self, reset_selection=False):
        available_columns = self.data_engine.get_available_training_columns(include_engineered=True)
        previous_selection = [] if reset_selection else self.data_engine.get_selected_training_columns()
        if reset_selection or not previous_selection:
            selected_columns = list(available_columns)
        else:
            selected_columns = [col for col in available_columns if col in previous_selection]

        self.data_engine.set_selected_training_columns(selected_columns)

        self.feature_selection_table.blockSignals(True)
        self.feature_selection_table.clearContents()
        self.feature_selection_table.setRowCount(len(available_columns))

        for row, column in enumerate(available_columns):
            use_item = QTableWidgetItem()
            use_item.setFlags(
                Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsUserCheckable
            )
            use_item.setCheckState(
                Qt.CheckState.Checked if column in selected_columns else Qt.CheckState.Unchecked
            )
            self.feature_selection_table.setItem(row, 0, use_item)

            name_item = QTableWidgetItem(column)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.feature_selection_table.setItem(row, 1, name_item)

            column_type = "합금 지표" if column in self.data_engine.engineered_feature_cols else "원본"
            type_item = QTableWidgetItem(column_type)
            type_item.setFlags(type_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.feature_selection_table.setItem(row, 2, type_item)

        self.feature_selection_table.blockSignals(False)
        self.feature_selection_table.resizeColumnsToContents()

        has_columns = bool(available_columns)
        self.select_all_features_btn.setEnabled(has_columns)
        self.clear_features_btn.setEnabled(has_columns)
        self.refresh_feature_selection_summary()

    def get_checked_feature_columns_from_table(self):
        selected_columns = []
        for row in range(self.feature_selection_table.rowCount()):
            use_item = self.feature_selection_table.item(row, 0)
            name_item = self.feature_selection_table.item(row, 1)
            if use_item and name_item and use_item.checkState() == Qt.CheckState.Checked:
                selected_columns.append(name_item.text())
        return selected_columns

    def refresh_feature_selection_summary(self):
        available_columns = self.data_engine.get_available_training_columns(include_engineered=True)
        selected_columns = self.data_engine.get_selected_training_columns()

        if not available_columns:
            self.feature_selection_status_label.setText(
                "먼저 전처리를 실행한 뒤, 이 탭에서 학습 컬럼을 선택해 주세요."
            )
            self.go_to_model_training_btn.setEnabled(False)
            return

        raw_count = sum(1 for col in selected_columns if col in self.data_engine.raw_feature_cols)
        engineered_count = sum(
            1 for col in selected_columns if col in self.data_engine.engineered_feature_cols
        )
        self.feature_selection_status_label.setText(
            f"전체 {len(available_columns)}개 중 {len(selected_columns)}개 컬럼이 선택되었습니다. "
            f"(원본 {raw_count}개, 합금 지표 {engineered_count}개)"
        )
        self.go_to_model_training_btn.setEnabled(self.preprocessing_ready and bool(selected_columns))

        if self.preprocessing_ready:
            if selected_columns:
                self.train_btn.setEnabled(True)
                self.training_data_status_label.setText(
                    f"전처리가 완료되었습니다. 선택한 {len(selected_columns)}개 컬럼으로 학습합니다."
                )
            else:
                self.train_btn.setEnabled(False)
                self.training_data_status_label.setText(
                    "전처리는 완료되었지만 아직 학습 컬럼이 선택되지 않았습니다."
                )

    def on_feature_selection_item_changed(self, item):
        if item.column() != 0:
            return
        self.data_engine.set_selected_training_columns(self.get_checked_feature_columns_from_table())
        self.refresh_feature_selection_summary()

    def select_all_feature_columns(self):
        self.feature_selection_table.blockSignals(True)
        for row in range(self.feature_selection_table.rowCount()):
            item = self.feature_selection_table.item(row, 0)
            if item:
                item.setCheckState(Qt.CheckState.Checked)
        self.feature_selection_table.blockSignals(False)
        self.data_engine.set_selected_training_columns(self.get_checked_feature_columns_from_table())
        self.refresh_feature_selection_summary()

    def clear_all_feature_columns(self):
        self.feature_selection_table.blockSignals(True)
        for row in range(self.feature_selection_table.rowCount()):
            item = self.feature_selection_table.item(row, 0)
            if item:
                item.setCheckState(Qt.CheckState.Unchecked)
        self.feature_selection_table.blockSignals(False)
        self.data_engine.set_selected_training_columns(self.get_checked_feature_columns_from_table())
        self.refresh_feature_selection_summary()

    def reset_preprocessing_state(self, keep_file=True):
        current_file = self.data_engine.file_path if keep_file else None
        self.preprocessing_ready = False
        self.data_engine.df = None
        self.data_engine.last_quality_report = {}
        self.data_engine.set_selected_training_columns([])
        if not keep_file:
            self.data_engine.set_file_path(None)

        self.train_btn.setEnabled(False)
        self.go_to_training_btn.setEnabled(False)
        if hasattr(self, "go_to_model_training_btn"):
            self.go_to_model_training_btn.setEnabled(False)
        if hasattr(self, "generate_features_btn"):
            self.generate_features_btn.setEnabled(False)
        self.metrics_label.setText("<b>모델 성능 요약:</b><br>- 예측 정확도: N/A<br>- 평균 오차: N/A")
        self.training_data_status_label.setText("")
        self.training_status_label.setText("")
        self.quality_summary_label.setText("전처리 결과 요약이 아직 없습니다.")
        self.domain_rule_label.setText("도메인 검증은 '오스테나이트 조성 기준'과 '고온 특성 기준' 두 부류로 나누어 범위를 확인합니다.")
        self.processed_preview_info_label.setText("전처리를 실행하면 처리 완료된 전체 데이터를 아래 표에서 확인할 수 있습니다.")
        self.processed_preview_table.clear()
        self.processed_preview_table.setRowCount(0)
        self.processed_preview_table.setColumnCount(0)
        self.engineered_preview_table.clear()
        self.engineered_preview_table.setRowCount(0)
        self.engineered_preview_table.setColumnCount(0)
        if hasattr(self, "feature_selection_table"):
            self.feature_selection_table.clearContents()
            self.feature_selection_table.setRowCount(0)
            self.select_all_features_btn.setEnabled(False)
            self.clear_features_btn.setEnabled(False)
            self.feature_selection_status_label.setText(
                "먼저 전처리를 실행한 뒤, 이 탭에서 학습 컬럼을 선택해 주세요."
            )
        self.render_training_placeholder()
        self.render_performance_placeholder()

        if keep_file and current_file:
            self.file_path_label.setText(f"파일: {os.path.basename(current_file)}")
            self.status_label.setText("상태: 전처리 설정을 초기화했습니다. 다시 전처리를 실행해 주세요.")
        else:
            self.file_path_label.setText("파일: 선택되지 않음")
            self.status_label.setText("상태: 학습용 데이터를 선택해 주세요.")

    def on_reset_preprocessing_clicked(self):
        self.missing_combo.setCurrentIndex(0)
        self.outlier_combo.setCurrentIndex(0)
        self.invalid_type_combo.setCurrentIndex(0)
        self.iqr_spin.setValue(1.5)
        self.feature_engineering_check.blockSignals(True)
        self.feature_engineering_check.setChecked(False)
        self.feature_engineering_check.blockSignals(False)
        if hasattr(self, "training_input_combo"):
            self.training_input_combo.setCurrentIndex(0)
        self.reset_preprocessing_state(keep_file=True)

    def on_select_file_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "데이터 파일 열기",
            "",
            "데이터 파일 (*.xls *.xlsx *.xlsm *.csv)",
        )
        if not file_path:
            return

        self.data_engine.set_file_path(file_path)
        self.data_engine.df = None
        self.data_engine.last_quality_report = {}
        self.file_path_label.setText(f"파일: {os.path.basename(file_path)}")
        self.status_label.setText("상태: 새 데이터 파일이 선택되었습니다. 1번 탭에서 전처리를 실행해 주세요.")
        self.quality_summary_label.setText("전처리 결과 요약이 아직 없습니다.")
        self.domain_rule_label.setText("조성, 열처리 온도, 시간, 이진값 등에 대해 '말이 되는 범위'를 확인합니다.")
        self.processed_preview_info_label.setText("전처리를 실행하면 처리 완료된 전체 데이터를 아래 표에서 확인할 수 있습니다.")
        self.processed_preview_table.clear()
        self.processed_preview_table.setRowCount(0)
        self.processed_preview_table.setColumnCount(0)
        self.metrics_label.setText("<b>모델 성능 요약:</b><br>- 예측 정확도: N/A<br>- 평균 오차: N/A")
        self.reset_preprocessing_state(keep_file=True)
        self.status_label.setText("상태: 새 데이터 파일이 선택되었습니다. 전처리를 실행해 주세요.")
        self._update_project_tree()

    def on_preprocess_clicked(self):
        if not self.data_engine.file_path or not os.path.exists(self.data_engine.file_path):
            self.status_label.setText("상태: 오류 - 먼저 데이터 파일을 선택해 주세요.")
            return

        self.apply_quality_settings_from_ui()
        self.data_engine.quality_options["feature_engineering"] = False
        self.feature_engineering_check.blockSignals(True)
        self.feature_engineering_check.setChecked(False)
        self.feature_engineering_check.blockSignals(False)
        self.status_label.setText("상태: 1차 데이터 정제 전처리를 실행 중입니다.")
        self.preprocess_btn.setEnabled(False)
        self.processed_preview_info_label.setText("새 설정으로 전처리 중입니다. 완료되면 표가 갱신됩니다.")
        self.processed_preview_table.clear()
        self.processed_preview_table.setRowCount(0)
        self.processed_preview_table.setColumnCount(0)
        self.engineered_preview_table.clear()
        self.engineered_preview_table.setRowCount(0)
        self.engineered_preview_table.setColumnCount(0)

        try:
            processed_df = self.data_engine.load_data(include_engineered=False)
            self.update_quality_summary_from_report(self.data_engine.last_quality_report)
            self.populate_processed_preview(processed_df)
            self.populate_feature_selection_table(reset_selection=True)
            self.preprocessing_ready = True
            self.status_label.setText("상태: 1차 데이터 정제 전처리가 완료되었습니다.")
            self.training_data_status_label.setText(f"전처리 완료 데이터 {len(processed_df)}행이 준비되었습니다. 이제 모델 학습을 시작할 수 있습니다.")
            self.training_status_label.setText("상태: 전처리 완료. 2번 탭에서 모델을 학습할 수 있습니다.")
            self.generate_features_btn.setEnabled(True)
            self.train_btn.setEnabled(True)
            self.go_to_training_btn.setEnabled(True)
            self.refresh_feature_selection_summary()
            self._update_project_tree()
            self._sb_status.setText("● 전처리 완료")
            self._sb_status.setStyleSheet("color: #E56020; font-size: 10px; font-family: 'Consolas', monospace; padding: 0 10px;")
        except Exception as exc:
            self.preprocessing_ready = False
            self.train_btn.setEnabled(False)
            self.go_to_training_btn.setEnabled(False)
            self.generate_features_btn.setEnabled(False)
            self.status_label.setText(f"상태: 전처리 오류 - {exc}")
        finally:
            self.preprocess_btn.setEnabled(True)

    def on_generate_features_clicked(self):
        try:
            if self.data_engine.df is None or self.data_engine.df.empty:
                self.status_label.setText("상태: 먼저 데이터 전처리를 실행해 주세요.")
                return

            self.status_label.setText("상태: 2차 합금 지표 생성 전처리를 실행 중입니다.")
            self.generate_features_btn.setEnabled(False)
            self.data_engine.generate_engineered_features_on_current_df()
            self.feature_engineering_check.blockSignals(True)
            self.feature_engineering_check.setChecked(True)
            self.feature_engineering_check.blockSignals(False)
            self.update_quality_summary_from_report(self.data_engine.last_quality_report)
            self.populate_processed_preview(self.data_engine.df)
            self.populate_feature_selection_table(reset_selection=False)
            self.status_label.setText("상태: 합금 지표 생성이 완료되었습니다.")
            self.training_status_label.setText("상태: 합금 지표 생성 완료. 선택한 학습 데이터 방식으로 모델을 학습할 수 있습니다.")
        except Exception as exc:
            self.status_label.setText(f"상태: 합금 지표 생성 오류 - {exc}")
        finally:
            self.generate_features_btn.setEnabled(True)

    def show_austenite_domain_dialog(self):
        self.show_domain_range_dialog("오스테나이트 조성 기준")

    def show_high_temp_domain_dialog(self):
        self.show_domain_range_dialog("고온 특성 기준")

    def show_domain_range_dialog(self, group_filter=None):
        dialog = QDialog(self)
        dialog.setWindowTitle("도메인 기준 설정")
        dialog.resize(760, 620)

        layout = QVBoxLayout(dialog)
        intro = QLabel("각 컬럼의 최소값과 최대값을 직접 설정할 수 있습니다. SSINA 기준을 반영해 오스테나이트 조성 기준과 고온 특성 기준 두 부류로 나누어 표시합니다.")
        intro.setWordWrap(True)
        intro.setStyleSheet("font-size: 13px; color: #34495e; padding-bottom: 4px;")
        layout.addWidget(intro)

        table = QTableWidget(dialog)
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(["부류", "컬럼", "최소값", "최대값", "근거"])
        domain_ranges = self.data_engine.get_domain_ranges()
        sorted_columns = sorted(
            [
                col
                for col in domain_ranges.keys()
                if group_filter is None or self.data_engine.get_domain_group(col) == group_filter
            ],
            key=lambda col: (self.data_engine.get_domain_group(col), col),
        )
        table.setRowCount(len(sorted_columns))
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)

        for row, column in enumerate(sorted_columns):
            lower_bound, upper_bound = domain_ranges[column]
            group = self.data_engine.get_domain_group(column)
            basis = self.data_engine.get_domain_basis(column)
            source = "사용자 지정" if column in self.data_engine.custom_ranges else basis
            group_item = QTableWidgetItem(group)
            group_item.setFlags(group_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(row, 0, group_item)
            name_item = QTableWidgetItem(column)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(row, 1, name_item)
            table.setItem(row, 2, QTableWidgetItem("" if lower_bound is None else str(lower_bound)))
            table.setItem(row, 3, QTableWidgetItem("" if upper_bound is None else str(upper_bound)))
            source_item = QTableWidgetItem(source)
            source_item.setFlags(source_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(row, 4, source_item)

        layout.addWidget(table, 1)

        button_row = QHBoxLayout()
        reset_btn = QPushButton("기본값으로 되돌리기")
        apply_btn = QPushButton("적용")
        close_btn = QPushButton("닫기")
        button_row.addWidget(reset_btn)
        button_row.addStretch()
        button_row.addWidget(apply_btn)
        button_row.addWidget(close_btn)
        layout.addLayout(button_row)

        def apply_ranges(reset_to_default=False):
            if reset_to_default:
                self.data_engine.reset_custom_domain_ranges()
                self.refresh_domain_range_status()
                self.domain_rule_label.setText("도메인 검증은 '오스테나이트 조성 기준'과 '고온 특성 기준' 두 부류로 나누어 범위를 확인합니다.")
                self.mark_preprocessing_dirty()
                dialog.accept()
                return

            custom_ranges = {}
            for row, column in enumerate(sorted_columns):
                min_item = table.item(row, 2)
                max_item = table.item(row, 3)
                min_text = min_item.text().strip() if min_item and min_item.text() else ""
                max_text = max_item.text().strip() if max_item and max_item.text() else ""

                try:
                    lower_bound = None if min_text == "" else float(min_text)
                    upper_bound = None if max_text == "" else float(max_text)
                except ValueError:
                    QMessageBox.warning(dialog, "입력 오류", f"{column} 값의 최소값 또는 최대값이 숫자가 아닙니다.")
                    return

                if lower_bound is not None and upper_bound is not None and lower_bound > upper_bound:
                    QMessageBox.warning(dialog, "입력 오류", f"{column} 값에서 최소값이 최대값보다 큽니다.")
                    return

                default_bounds = self.data_engine.default_domain_ranges.get(column)
                current_bounds = (lower_bound, upper_bound)
                if default_bounds is None or current_bounds != default_bounds:
                    custom_ranges[column] = current_bounds

            self.data_engine.set_custom_domain_ranges(custom_ranges)
            self.refresh_domain_range_status()
            self.domain_rule_label.setText("도메인 검증은 '오스테나이트 조성 기준'과 '고온 특성 기준' 두 부류의 범위를 기준으로 비정상 값을 확인합니다.")
            self.mark_preprocessing_dirty()
            QMessageBox.information(dialog, "적용 완료", "도메인 기준이 저장되었습니다. 전처리를 다시 실행하면 반영됩니다.")
            dialog.accept()

        reset_btn.clicked.connect(lambda: apply_ranges(True))
        apply_btn.clicked.connect(apply_ranges)
        close_btn.clicked.connect(dialog.reject)
        dialog.exec()

    def on_train_clicked(self):
        if not self.data_engine.file_path or not os.path.exists(self.data_engine.file_path):
            self.training_status_label.setText("상태: 오류 - 먼저 데이터 파일을 선택해 주세요.")
            return

        if not self.preprocessing_ready:
            self.training_status_label.setText("상태: 1번 탭에서 전처리를 먼저 실행해 주세요.")
            return

        selected_columns = self.data_engine.get_selected_training_columns()
        if not selected_columns:
            self.training_status_label.setText("상태: 오류 - 학습할 컬럼을 하나 이상 선택해 주세요.")
            self.tabs.setCurrentIndex(1)
            return

        self.apply_quality_settings_from_ui()
        self.train_btn.setEnabled(False)
        self.training_status_label.setText(
            f"상태: 학습 준비 중입니다. 선택한 컬럼 {len(selected_columns)}개만 사용합니다."
        )
        self.metrics_label.setText("<b>모델 성능 요약:</b><br>- 계산 중...")

        model_map = {0: "RF", 1: "GBM", 2: "MLP", 3: "TFP"}
        self.model_type = model_map.get(self.model_combo.currentIndex(), "RF")

        self.thread = TrainingThread(self.data_engine, model_type=self.model_type, max_iter=self.iter_spin.value())
        self.thread.progress.connect(lambda text: self.training_status_label.setText(f"상태: {text}"))
        self.thread.finished.connect(self.on_training_finished)
        self.thread.start()

    def on_training_finished(self, results):
        self.train_btn.setEnabled(True)
        if isinstance(results, str):
            self.training_status_label.setText(f"상태: 오류 발생 - {results}")
            return

        self.model_engine = results["model"]
        self.training_status_label.setText(f"상태: {self.model_type} 모델 학습이 완료되었습니다.")
        self.update_active_model_display()
        self.update_quality_summary_from_report(results.get("quality_report", {}))

        metrics = results["metrics"]
        r2_avg = float(np.mean(metrics["r2"]))
        mae_avg = float(np.mean(metrics["mae"]))
        self.last_r2_avg = round(r2_avg, 4)
        acc_text = "매우 높음" if r2_avg > 0.9 else "높음" if r2_avg > 0.8 else "보통"

        self.metrics_label.setText(
            f"<b>종합 모델 성능 요약:</b><br>- 평균 예측 정확도(R2): <b>{r2_avg * 100:.1f}% ({acc_text})</b><br>- 평균 오차(MAE): <b>{mae_avg:.2f}</b>"
        )

        self.canvas.axes.clear()
        target_names = ["Yield Stress", "UTS", "Elongation", "Area Red."]
        r2_scores = metrics["r2"]
        colors = ["#3498db" if score > 0.8 else "#f1c40f" if score > 0.6 else "#e74c3c" for score in r2_scores]
        bars = self.canvas.axes.bar(target_names, r2_scores, color=colors)
        self.canvas.axes.set_ylim(0, 1.1)
        self.canvas.axes.set_ylabel("정확도 (R2 Score)")
        for bar in bars:
            height = bar.get_height()
            self.canvas.axes.text(bar.get_x() + bar.get_width() / 2.0, height + 0.02, f"{height:.2f}", ha="center", va="bottom", fontsize=9)
        name_map = {"RF": "Random Forest", "GBM": "Gradient Boosting", "MLP": "Neural Network", "TFP": "TFP"}
        self.canvas.axes.set_title(f"모델별 특성 예측 정확도 ({name_map.get(self.model_type, self.model_type)})")
        self.canvas.draw()
        self.render_performance_results(results)

        # [AUTO SAVE] 학습 완료 시 auto_save/ 폴더에 자동 저장
        self.auto_save_workspace()
        self._update_project_tree()
        self._sb_status.setText("● 학습 완료")
        self._sb_status.setStyleSheet("color: #4CAF50; font-size: 10px; font-family: 'Consolas', monospace; padding: 0 10px;")

        # [LOG] 학습 완료 로그 기록
        self.append_log({
            "type": "학습",
            "model": self.model_type,
            "data_file": os.path.basename(self.data_engine.file_path or ""),
            "r2_avg": round(float(np.mean(metrics["r2"])), 4),
            "mae_avg": round(float(np.mean(metrics["mae"])), 4),
            "r2_per_target": [round(float(v), 4) for v in metrics["r2"]],
            "mae_per_target": [round(float(v), 4) for v in metrics["mae"]],
        })

    def render_performance_results(self, results):
        self.perf_canvas.figure.clear()
        axes = self.perf_canvas.figure.subplots(2, 2)
        y_test = results["y_test"].values
        y_pred = results["y_pred"]
        target_names = ["Yield Stress (MPa)", "UTS (MPa)", "Elongation (%)", "Area Reduction (%)"]
        colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]

        for index, ax in enumerate(axes.flatten()):
            ax.scatter(y_test[:, index], y_pred[:, index], alpha=0.55, color=colors[index], s=18)
            all_data = np.concatenate([y_test[:, index], y_pred[:, index]])
            min_val, max_val = all_data.min(), all_data.max()
            ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.7, lw=1)
            ax.set_title(target_names[index], fontsize=10, fontweight="bold")
            ax.set_xlabel("실제값", fontsize=9)
            ax.set_ylabel("예측값", fontsize=9)
            ax.grid(True, linestyle=":", alpha=0.6)

        self.perf_canvas.figure.tight_layout()
        self.perf_canvas.draw()

    def on_predict_clicked(self):
        if not self.model_engine:
            self.result_display.setText("Please train or load the model first!")
            return

        try:
            input_dict = {key: widget.text() for key, widget in self.inputs.items()}
            scaled_input = self.data_engine.get_inference_data(input_dict)
            mean_scaled, std_scaled = self.model_engine.predict(scaled_input.astype(np.float32))

            mean = self.data_engine.scaler_y.inverse_transform(mean_scaled)[0]
            std = std_scaled[0] * self.data_engine.scaler_y.scale_

            result_text = (
                f"<b>[강도 예측 결과]</b><br>"
                f"0.2% Yield Stress: <b>{mean[0]:.1f} ± {std[0]:.1f} MPa</b><br>"
                f"UTS: <b>{mean[1]:.1f} ± {std[1]:.1f} MPa</b><br><br>"
                f"<b>[연성 예측 결과]</b><br>"
                f"Elongation: <b>{mean[2]:.1f} ± {std[2]:.1f} %</b><br>"
                f"Area Reduction: <b>{mean[3]:.1f} ± {std[3]:.1f} %</b>"
            )
            self.result_display.setText(result_text)

            self.prediction_canvas.axes.clear()
            labels = ["Yield", "UTS", "Elong.", "Area Red."]
            x = np.arange(len(labels))

            stress_vals = [mean[0], mean[1], 0, 0]
            stress_errs = [1.96 * std[0], 1.96 * std[1], 0, 0]
            self.prediction_canvas.axes.bar(x[:2], stress_vals[:2], yerr=stress_errs[:2], capsize=10, color=["#3498db", "#e74c3c"])
            self.prediction_canvas.axes.set_ylabel("Stress (MPa)", color="#3498db")

            ax2 = self.prediction_canvas.axes.twinx()
            ax2.clear()
            duct_vals = [0, 0, mean[2], mean[3]]
            duct_errs = [0, 0, 1.96 * std[2], 1.96 * std[3]]
            ax2.bar(x[2:], duct_vals[2:], yerr=duct_errs[2:], capsize=10, color=["#2ecc71", "#f39c12"])
            ax2.set_ylabel("Percentage (%)", color="#2ecc71")

            self.prediction_canvas.axes.set_xticks(x)
            self.prediction_canvas.axes.set_xticklabels(labels)
            self.prediction_canvas.axes.set_title("Predicted Properties")
            self.prediction_canvas.draw()

            # [AUTO SAVE] 예측 완료 시 auto_save 폴더에 예측 그래프 저장
            auto_folder = os.path.join("workspaces", "auto_save")
            if os.path.exists(auto_folder):
                self.prediction_canvas.fig.savefig(
                    os.path.join(auto_folder, "prediction.png"), dpi=200, bbox_inches="tight")

            # [LOG] 예측 실행 로그 기록
            self.append_log({
                "type": "예측",
                "model": self.model_type,
                "inputs": {k: v.text() for k, v in self.inputs.items()},
                "results": {
                    "yield_stress": round(float(mean[0]), 2),
                    "uts": round(float(mean[1]), 2),
                    "elongation": round(float(mean[2]), 2),
                    "area_reduction": round(float(mean[3]), 2),
                },
            })
        except Exception as exc:
            self.result_display.setText(f"Error during prediction: {exc}")

    def update_active_model_display(self):
        name_map = {"RF": "Random Forest", "GBM": "Gradient Boosting", "MLP": "Neural Network", "TFP": "TFP"}
        self.active_model_info.setText(f"현재 예측 모델: {name_map.get(self.model_type, self.model_type)}")
        self.active_model_info.setStyleSheet(
            "background-color: #d4efdf; padding: 10px; border: 1px solid #27ae60; border-radius: 5px; font-weight: bold; margin-bottom: 10px;"
        )

    # ================================================================
    # [AUTO SAVE] 학습 완료 시 workspaces/auto_save/ 폴더에 자동 저장 (1개만 유지, 덮어씀)
    # ================================================================
    def auto_save_workspace(self):
        import shutil
        folder = os.path.join("workspaces", "auto_save")
        if os.path.exists(folder):
            shutil.rmtree(folder)  # [AUTO SAVE] 오래된 auto_save 폴더 삭제
        os.makedirs(folder)
        state = {
            "file_path": self.data_engine.file_path,
            "model_combo_index": self.model_combo.currentIndex(),
            "max_iter": self.iter_spin.value(),
            "inputs": {k: v.text() for k, v in self.inputs.items()},
            # [AUTO SAVE] 전처리 설정값 저장
            "preprocessing": {
                "missing_combo": self.missing_combo.currentIndex(),
                "outlier_combo": self.outlier_combo.currentIndex(),
                "invalid_type_combo": self.invalid_type_combo.currentIndex(),
                "iqr_spin": self.iqr_spin.value(),
                "training_input_combo": self.training_input_combo.currentIndex(),
                "preprocessing_ready": self.preprocessing_ready,
            },
        }
        with open(os.path.join(folder, "state.json"), "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        self.canvas.fig.savefig(os.path.join(folder, "training.png"), dpi=200, bbox_inches="tight")
        self.perf_canvas.figure.savefig(os.path.join(folder, "performance.png"), dpi=200, bbox_inches="tight")
        # [AUTO SAVE] 전처리 결과 CSV 저장
        pre_df = self.data_engine.get_preprocessed_display_df()
        if not pre_df.empty:
            pre_df.to_csv(os.path.join(folder, "preprocessed_data.csv"), index=False, encoding="utf-8-sig")
        eng_df = self.data_engine.get_engineered_display_df()
        if not eng_df.empty:
            eng_df.to_csv(os.path.join(folder, "engineered_data.csv"), index=False, encoding="utf-8-sig")
        # [AUTO SAVE] 예측 그래프는 예측 실행 시점에 별도 저장됨
    # ================================================================

    # ================================================================
    # [LOG] workspaces/log.json 에 로그 항목 추가
    # ================================================================
    def append_log(self, entry):
        ws_dir = "workspaces"
        if not os.path.exists(ws_dir):
            os.makedirs(ws_dir)
        log_path = os.path.join(ws_dir, "log.json")
        logs = []
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                logs = json.load(f)
        entry["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logs.append(entry)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
    # ================================================================

    # ================================================================
    # [WORKSPACE] 선택한 분석 저장 삭제 (폴더 단위 삭제)
    # ================================================================
    def delete_workspace(self):
        import shutil
        name = self.ws_combo.currentText()
        if not name:
            self.status_label.setText("상태: 삭제할 분석 저장을 선택해 주세요")
            return
        reply = QMessageBox.question(self, "삭제 확인",
            f"'{name}' 분석 저장을 삭제하시겠습니까?\n(폴더 전체가 삭제됩니다)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.No:
            return
        folder = os.path.join("workspaces", name)
        if os.path.exists(folder):
            shutil.rmtree(folder)  # [WORKSPACE] 폴더 단위로 삭제
        self.refresh_workspace_list()
        self.status_label.setText(f"상태: 분석 저장 '{name}' 삭제 완료")
    # ================================================================

    # ================================================================
    # [WORKSPACE] workspaces/ 내 폴더 목록을 드롭다운에 갱신 (auto_save 제외)
    # ================================================================
    def refresh_workspace_list(self):
        ws_dir = "workspaces"
        self.ws_combo.clear()
        if os.path.exists(ws_dir):
            names = sorted([d for d in os.listdir(ws_dir)
                            if os.path.isdir(os.path.join(ws_dir, d)) and d != "auto_save"])
            self.ws_combo.addItems(names)
        if hasattr(self, "ws_table"):
            self.refresh_workspace_table()
    # ================================================================

    def refresh_workspace_table(self):
        ws_dir = "workspaces"
        self.ws_table.setRowCount(0)
        if not os.path.exists(ws_dir):
            return
        names = sorted([d for d in os.listdir(ws_dir)
                        if os.path.isdir(os.path.join(ws_dir, d)) and d != "auto_save"])
        model_name_map = {"RF": "Random Forest", "GBM": "Gradient Boosting", "MLP": "Neural Network", "TFP": "TFP"}

        for row, name in enumerate(names):
            self.ws_table.insertRow(row)
            folder = os.path.join(ws_dir, name)
            state_path = os.path.join(folder, "state.json")
            state = {}
            if os.path.exists(state_path):
                with open(state_path, "r", encoding="utf-8") as f:
                    state = json.load(f)

            # 이름
            self.ws_table.setItem(row, 0, QTableWidgetItem(name))

            # 모델
            model_idx = state.get("model_combo_index", -1)
            model_keys = ["RF", "GBM", "MLP", "TFP"]
            model_key = model_keys[model_idx] if 0 <= model_idx < len(model_keys) else "-"
            self.ws_table.setItem(row, 1, QTableWidgetItem(model_name_map.get(model_key, "-")))

            # 데이터 파일
            fp = state.get("file_path") or ""
            self.ws_table.setItem(row, 2, QTableWidgetItem(os.path.basename(fp) if fp else "-"))

            # 저장 날짜 / R2 (각 워크스페이스 state.json 기준)
            saved_date = state.get("saved_date", "-")
            r2_val = state.get("r2_avg")
            r2_text = f"{r2_val * 100:.1f}%" if r2_val is not None else "-"
            self.ws_table.setItem(row, 3, QTableWidgetItem(saved_date))
            self.ws_table.setItem(row, 4, QTableWidgetItem(r2_text))

    def _on_ws_table_clicked(self, row, _):
        """행 클릭 시 하단 상세 패널에 썸네일 + 정보 표시"""
        name_item = self.ws_table.item(row, 0)
        if not name_item:
            return
        name = name_item.text()
        folder = os.path.join("workspaces", name)

        # 썸네일 (training.png)
        thumb_path = os.path.join(folder, "training.png")
        if os.path.exists(thumb_path):
            pix = QPixmap(thumb_path).scaled(
                200, 120,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.ws_detail_thumb.setPixmap(pix)
            self._ws_thumb_full_path = thumb_path
        else:
            self.ws_detail_thumb.clear()
            self.ws_detail_thumb.setText("없음")
            self._ws_thumb_full_path = None

        # 상세 성능 썸네일 (performance.png)
        perf_path = os.path.join(folder, "performance.png")
        if os.path.exists(perf_path):
            pix2 = QPixmap(perf_path).scaled(
                200, 120,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.ws_detail_perf_thumb.setPixmap(pix2)
            self._ws_perf_thumb_full_path = perf_path
        else:
            self.ws_detail_perf_thumb.clear()
            self.ws_detail_perf_thumb.setText("없음")
            self._ws_perf_thumb_full_path = None

        # 상세 정보 텍스트
        state = {}
        state_path = os.path.join(folder, "state.json")
        if os.path.exists(state_path):
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
        model_keys = ["RF", "GBM", "MLP", "TFP"]
        model_name_map = {"RF": "Random Forest", "GBM": "Gradient Boosting", "MLP": "Neural Network", "TFP": "TFP"}
        model_idx = state.get("model_combo_index", -1)
        model_key = model_keys[model_idx] if 0 <= model_idx < len(model_keys) else "-"
        fp = os.path.basename(state.get("file_path") or "") or "-"
        max_iter = state.get("max_iter", "-")
        pre = state.get("preprocessing", {})
        missing_labels = ["평균값", "중앙값", "KNN", "행 제거"]
        outlier_labels = ["범위 보정", "이상치 제거", "플래그"]
        missing_txt = missing_labels[pre.get("missing_combo", 0)] if pre else "-"
        outlier_txt = outlier_labels[pre.get("outlier_combo", 0)] if pre else "-"
        iqr_txt = str(pre.get("iqr_spin", "-")) if pre else "-"
        has_pre_csv = os.path.exists(os.path.join(folder, "preprocessed_data.csv"))
        has_eng_csv = os.path.exists(os.path.join(folder, "engineered_data.csv"))

        info = (
            f"<b>이름:</b> {name}<br>"
            f"<b>모델:</b> {model_name_map.get(model_key, '-')}<br>"
            f"<b>데이터 파일:</b> {fp}<br>"
            f"<b>최대 반복 횟수:</b> {max_iter}<br><br>"
            f"<b>[전처리 설정]</b><br>"
            f"누락값 처리: {missing_txt}<br>"
            f"이상치 처리: {outlier_txt}<br>"
            f"IQR 민감도: {iqr_txt}<br><br>"
            f"<b>[저장된 파일]</b><br>"
            f"전처리 결과 CSV: {'✔' if has_pre_csv else '✘'}<br>"
            f"합금 지표 CSV: {'✔' if has_eng_csv else '✘'}"
        )
        self.ws_detail_info.setText(info)

    def _show_full_graph_dialog(self, path):
        """그래프 원본을 크게 보여주는 다이얼로그 (확대/축소 + 휠 줌)"""
        if not path or not os.path.exists(path):
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("그래프 크게 보기")
        dialog.resize(1100, 860)
        layout = QVBoxLayout(dialog)

        orig_pix = QPixmap(path)
        init_zoom = min(1040 / orig_pix.width(), 780 / orig_pix.height(), 1.0) if orig_pix.width() > 0 else 1.0
        zoom = [init_zoom]

        scroll = QScrollArea()
        scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img_label = QLabel()
        img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scroll.setWidget(img_label)
        layout.addWidget(scroll)

        def render():
            w = int(orig_pix.width() * zoom[0])
            h = int(orig_pix.height() * zoom[0])
            scaled = orig_pix.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            img_label.setPixmap(scaled)
            img_label.resize(scaled.width(), scaled.height())

        render()

        def zoom_in():
            zoom[0] = min(zoom[0] * 1.25, 8.0); render()
        def zoom_out():
            zoom[0] = max(zoom[0] * 0.8, init_zoom); render()
        def zoom_reset():
            zoom[0] = 1.0; render()

        scroll.wheelEvent = lambda e: zoom_in() if e.angleDelta().y() > 0 else zoom_out()

        btn_row = QHBoxLayout()
        for label, fn, color in [("확대 (+)", zoom_in, "#2980b9"), ("원래 크기", zoom_reset, "#27ae60")]:
            b = QPushButton(label)
            b.setFixedWidth(90)
            b.setStyleSheet(f"background-color: {color}; color: white; font-weight: bold; padding: 5px;")
            b.clicked.connect(fn)
            btn_row.addWidget(b)
        btn_row.addStretch()
        close_btn = QPushButton("닫기")
        close_btn.setFixedWidth(90)
        close_btn.setStyleSheet("background-color: #7f8c8d; color: white; font-weight: bold; padding: 5px;")
        close_btn.clicked.connect(dialog.close)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.destroyed.connect(lambda: self._open_dialogs.remove(dialog) if dialog in self._open_dialogs else None)
        self._open_dialogs.append(dialog)
        dialog.show()

    def _on_thumb_clicked(self, _):
        """학습 그래프 썸네일 클릭 시 크게 보기"""
        self._show_full_graph_dialog(self._ws_thumb_full_path)

    def _on_perf_thumb_clicked(self, _):
        """상세 성능 그래프 썸네일 클릭 시 크게 보기"""
        self._show_full_graph_dialog(self._ws_perf_thumb_full_path)

    def _on_compare_clicked(self):
        """선택된 행(최대 3개)의 그래프를 나란히 비교하는 다이얼로그"""
        selected_rows = list({idx.row() for idx in self.ws_table.selectedIndexes()})
        if len(selected_rows) < 2:
            QMessageBox.information(self, "비교", "비교할 분석을 2개 이상 선택해 주세요.\n(Ctrl+클릭으로 여러 행 선택)")
            return
        if len(selected_rows) > 3:
            QMessageBox.warning(self, "비교", "최대 3개까지만 비교할 수 있습니다.")
            return

        names = [self.ws_table.item(r, 0).text() for r in selected_rows if self.ws_table.item(r, 0)]
        n = len(names)

        dialog = QDialog(self)
        dialog.setWindowTitle(f"분석 비교 — {' vs '.join(names)}")
        dialog.resize(500 * n, 900)
        outer = QVBoxLayout(dialog)

        # 원본 pixmap + label 쌍 보관 (줌 적용용)
        all_pairs = []  # [(orig_pix, img_lbl), ...]
        init_zoom_cmp = 0.5
        zoom = [init_zoom_cmp]

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        scroll_area.setWidget(content_widget)
        outer.addWidget(scroll_area)

        # 그래프 종류별로 행 구성: 학습 / 상세 성능
        for graph_file, graph_label in [("training.png", "학습 그래프"), ("performance.png", "상세 성능")]:
            section_lbl = QLabel(f"▶ {graph_label}")
            section_lbl.setStyleSheet("font-weight: bold; font-size: 13px; color: #2c3e50; margin-top: 8px;")
            content_layout.addWidget(section_lbl)

            row_layout = QHBoxLayout()
            for name in names:
                col = QVBoxLayout()
                name_lbl = QLabel(name)
                name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                name_lbl.setStyleSheet("font-size: 11px; font-weight: bold; color: #8e44ad;")
                col.addWidget(name_lbl)

                img_lbl = QLabel()
                img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                img_lbl.setStyleSheet("border: 1px solid #dde1e6; background: white;")
                path = os.path.join("workspaces", name, graph_file)
                if os.path.exists(path):
                    orig_pix = QPixmap(path)
                    all_pairs.append((orig_pix, img_lbl))
                else:
                    img_lbl.setText("그래프 없음")
                    img_lbl.setFixedSize(380, 260)
                col.addWidget(img_lbl)
                row_layout.addLayout(col)
            content_layout.addLayout(row_layout)

        def render_all():
            for orig_pix, img_lbl in all_pairs:
                w = int(orig_pix.width() * zoom[0])
                h = int(orig_pix.height() * zoom[0])
                scaled = orig_pix.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                img_lbl.setPixmap(scaled)
                img_lbl.resize(scaled.width(), scaled.height())

        render_all()

        def zoom_in():
            zoom[0] = min(zoom[0] * 1.25, 8.0); render_all()
        def zoom_out():
            zoom[0] = max(zoom[0] * 0.8, init_zoom_cmp); render_all()
        def zoom_reset():
            zoom[0] = 0.5; render_all()

        scroll_area.wheelEvent = lambda e: zoom_in() if e.angleDelta().y() > 0 else zoom_out()

        btn_row = QHBoxLayout()
        for label, fn, color in [("확대 (+)", zoom_in, "#2980b9"), ("원래 크기", zoom_reset, "#27ae60")]:
            b = QPushButton(label)
            b.setFixedWidth(90)
            b.setStyleSheet(f"background-color: {color}; color: white; font-weight: bold; padding: 5px;")
            b.clicked.connect(fn)
            btn_row.addWidget(b)
        btn_row.addStretch()
        close_btn = QPushButton("닫기")
        close_btn.setFixedWidth(90)
        close_btn.setStyleSheet("background-color: #7f8c8d; color: white; font-weight: bold; padding: 5px;")
        close_btn.clicked.connect(dialog.close)
        btn_row.addWidget(close_btn)
        outer.addLayout(btn_row)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.destroyed.connect(lambda: self._open_dialogs.remove(dialog) if dialog in self._open_dialogs else None)
        self._open_dialogs.append(dialog)
        dialog.show()

    def _load_selected_ws(self):
        """하단 불러오기 버튼 클릭 시 선택된 행 불러오기"""
        selected = self.ws_table.selectedItems()
        if not selected:
            return
        row = self.ws_table.currentRow()
        self._on_ws_table_double_clicked(row, 0)

    def _on_ws_table_double_clicked(self, row, _):
        name_item = self.ws_table.item(row, 0)
        if not name_item:
            return
        name = name_item.text()
        reply = QMessageBox.question(self, "불러오기 확인",
            f"'{name}' 분석을 불러오시겠습니까?\n현재 작업 중인 내용이 변경될 수 있습니다.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.No:
            return
        self.ws_combo.setCurrentText(name)
        self.load_workspace()
        self.tabs.setCurrentIndex(0)

    # ================================================================
    # [WORKSPACE] 이름 입력 후 저장 → workspaces/{이름}/ 폴더에 저장
    # ================================================================
    def save_workspace(self):
        name = self.ws_name_input.text().strip()
        if not name:
            self.status_label.setText("상태: 분석 저장 이름을 입력해 주세요")
            return
        folder = os.path.join("workspaces", name)
        if os.path.exists(folder):
            reply = QMessageBox.question(self, "덮어쓰기 확인",
                f"'{name}' 분석 저장이 이미 존재합니다.\n덮어쓰시겠습니까?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No:
                return
        else:
            os.makedirs(folder)
        # [WORKSPACE] 그래프 PNG 저장
        self.canvas.fig.savefig(os.path.join(folder, "training.png"), dpi=200, bbox_inches="tight")
        self.perf_canvas.figure.savefig(os.path.join(folder, "performance.png"), dpi=200, bbox_inches="tight")
        self.prediction_canvas.fig.savefig(os.path.join(folder, "prediction.png"), dpi=200, bbox_inches="tight")
        state = {
            "file_path": self.data_engine.file_path,
            "model_combo_index": self.model_combo.currentIndex(),
            "max_iter": self.iter_spin.value(),
            "inputs": {k: v.text() for k, v in self.inputs.items()},
            "saved_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "r2_avg": self.last_r2_avg,
            # [WORKSPACE] 전처리 설정값 저장
            "preprocessing": {
                "missing_combo": self.missing_combo.currentIndex(),
                "outlier_combo": self.outlier_combo.currentIndex(),
                "invalid_type_combo": self.invalid_type_combo.currentIndex(),
                "iqr_spin": self.iqr_spin.value(),
                "training_input_combo": self.training_input_combo.currentIndex(),
                "preprocessing_ready": self.preprocessing_ready,
            },
        }
        with open(os.path.join(folder, "state.json"), "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        # [WORKSPACE] 전처리 결과 / 합금 지표 결과 CSV 저장
        pre_df = self.data_engine.get_preprocessed_display_df()
        if not pre_df.empty:
            pre_df.to_csv(os.path.join(folder, "preprocessed_data.csv"), index=False, encoding="utf-8-sig")
        eng_df = self.data_engine.get_engineered_display_df()
        if not eng_df.empty:
            eng_df.to_csv(os.path.join(folder, "engineered_data.csv"), index=False, encoding="utf-8-sig")
        self.refresh_workspace_list()  # [WORKSPACE] 저장 후 드롭다운 목록 갱신
        self.status_label.setText(f"상태: 분석 저장 '{name}' 저장 완료")
    # ================================================================

    # ================================================================
    # [WORKSPACE] 드롭다운 선택 후 불러오기 → workspaces/{이름}/ 폴더에서 복원
    # ================================================================
    def load_workspace(self):
        name = self.ws_combo.currentText()
        if not name:
            self.status_label.setText("상태: 불러올 분석 저장을 선택해 주세요")
            return
        folder = os.path.join("workspaces", name)
        state_path = os.path.join(folder, "state.json")
        if not os.path.exists(state_path):
            self.status_label.setText("상태: 분석 저장 파일을 찾을 수 없습니다")
            return
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        saved_file = state.get("file_path")
        if saved_file and os.path.exists(saved_file):
            self.data_engine.set_file_path(saved_file)
            self.file_path_label.setText(f"파일: {os.path.basename(saved_file)}")
        self.model_combo.setCurrentIndex(state.get("model_combo_index", 0))
        self.iter_spin.setValue(state.get("max_iter", 2000))
        for k, v in state.get("inputs", {}).items():
            if k in self.inputs:
                self.inputs[k].setText(v)

        # [WORKSPACE] 전처리 설정값 복원
        pre = state.get("preprocessing", {})
        if pre:
            self.missing_combo.blockSignals(True)
            self.outlier_combo.blockSignals(True)
            self.invalid_type_combo.blockSignals(True)
            self.iqr_spin.blockSignals(True)
            self.training_input_combo.blockSignals(True)
            self.missing_combo.setCurrentIndex(pre.get("missing_combo", 0))
            self.outlier_combo.setCurrentIndex(pre.get("outlier_combo", 0))
            self.invalid_type_combo.setCurrentIndex(pre.get("invalid_type_combo", 0))
            self.iqr_spin.setValue(pre.get("iqr_spin", 1.5))
            self.training_input_combo.setCurrentIndex(pre.get("training_input_combo", 0))
            self.preprocessing_ready = pre.get("preprocessing_ready", False)
            self.train_btn.setEnabled(self.preprocessing_ready)
            self.go_to_training_btn.setEnabled(self.preprocessing_ready)
            self.missing_combo.blockSignals(False)
            self.outlier_combo.blockSignals(False)
            self.invalid_type_combo.blockSignals(False)
            self.iqr_spin.blockSignals(False)
            self.training_input_combo.blockSignals(False)

        # [WORKSPACE] 전처리 결과 테이블 복원 (CSV)
        pre_csv = os.path.join(folder, "preprocessed_data.csv")
        if os.path.exists(pre_csv):
            pre_df = pd.read_csv(pre_csv, encoding="utf-8-sig")
            self.populate_processed_preview(pre_df)

        # [WORKSPACE] 합금 지표 결과 테이블 복원 (CSV)
        eng_csv = os.path.join(folder, "engineered_data.csv")
        if os.path.exists(eng_csv):
            eng_df = pd.read_csv(eng_csv, encoding="utf-8-sig")
            self.engineered_preview_table.clear()
            self.engineered_preview_table.setRowCount(len(eng_df))
            self.engineered_preview_table.setColumnCount(len(eng_df.columns))
            self.engineered_preview_table.setHorizontalHeaderLabels([str(c) for c in eng_df.columns])
            for r, (_, row) in enumerate(eng_df.iterrows()):
                for c, val in enumerate(row):
                    text = "" if pd.isna(val) else f"{float(val):.4g}" if isinstance(val, (int, float, np.integer, np.floating)) else str(val)
                    item = QTableWidgetItem(text)
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.engineered_preview_table.setItem(r, c, item)
            self.engineered_preview_table.resizeColumnsToContents()

        # [WORKSPACE] 그래프 이미지 복원 (폴더 내 PNG)
        for img_file, canvas_fn in [
            ("training.png",    lambda img: (self.canvas.axes.clear(), self.canvas.axes.imshow(img), self.canvas.axes.axis("off"), self.canvas.draw())),
            ("performance.png", lambda img: (self.perf_canvas.figure.clear(), self.perf_canvas.figure.add_subplot(111).imshow(img), self.perf_canvas.figure.axes[0].axis("off"), self.perf_canvas.draw())),
            ("prediction.png",  lambda img: (self.prediction_canvas.axes.clear(), self.prediction_canvas.axes.imshow(img), self.prediction_canvas.axes.axis("off"), self.prediction_canvas.draw())),
        ]:
            path = os.path.join(folder, img_file)
            if os.path.exists(path):
                canvas_fn(plt.imread(path))
        self.status_label.setText(f"상태: 분석 저장 '{name}' 복원 완료")
    # ================================================================

    def resizeEvent(self, event):
        super().resizeEvent(event)


    def show_quality_help(self):
        help_text = """
        <h2>데이터 전처리 도움말</h2>
        <p>전처리 탭에서는 누락값, 이상치, 형식 오류, 도메인 범위를 학습 전에 먼저 정리합니다.</p>
        <h3>권장 순서</h3>
        <p>파일 선택 → 도메인 기준 설정 → 데이터 품질 처리 설정 → 전처리 실행 → 합금 지표 생성 → 결과 확인</p>
        <h3>추천 시작값</h3>
        <p>누락값 처리: <b>중앙값으로 채우기</b><br>이상치 처리: <b>감지 범위로 보정</b><br>형식 검증: <b>잘못된 값을 NaN으로 변환</b><br>이상치 민감도: <b>1.5</b></p>
        <h3>합금 지표 설명</h3>
        <p><b>Ni 당량 (Ni_eq)</b>: 오스테나이트 안정성을 보는 지표입니다.<br><b>Cr 당량 (Cr_eq)</b>: 페라이트 형성 경향을 보는 지표입니다.<br><b>Cr/Ni 비율</b>: 조직 균형을 보는 지표입니다.<br><b>침입형 원소 합 (C+N)</b>: 고용강화와 고온 강도에 영향을 주는 지표입니다.</p>
        <h3>결과 확인</h3>
        <p>전처리 또는 합금 지표 생성이 끝나면 오른쪽 표에서 전체 데이터를 바로 확인할 수 있습니다.</p>
        """
        self.show_help_dialog("전처리 도움말", help_text)

    def show_model_training_help(self):
        help_text = """
        <h2>모델 학습 도움말</h2>
        <p>모델 학습 탭에서는 전처리 완료 데이터를 이용해 모델을 학습합니다.</p>
        <h3>모델 선택 가이드</h3>
        <p><b>Random Forest</b>: 처음 시작할 때 가장 무난합니다.<br><b>Gradient Boosting</b>: 비교용으로 좋습니다.<br><b>Neural Network</b>: 데이터가 충분할 때 시도해 볼 수 있습니다.<br><b>TFP</b>: 불확실성까지 함께 보고 싶을 때 사용합니다.</p>
        <h3>학습 데이터 선택</h3>
        <p><b>데이터 정제 + 합금 지표</b>: 가장 기본 추천 방식입니다.<br><b>데이터 정제만</b>: 원본 정제 변수만 사용합니다.<br><b>합금 지표만</b>: 생성된 합금 지표만 사용합니다.</p>
        <h3>주의 사항</h3>
        <p>전처리 설정을 바꿨다면 최신 설정을 반영하기 위해 전처리를 다시 실행한 뒤 학습해 주세요.</p>
        """
        self.show_help_dialog("모델 학습 도움말", help_text)

    def show_help_dialog(self, title, html_text):
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(820, 640)

        layout = QVBoxLayout(dialog)
        browser = QTextBrowser(dialog)
        browser.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        browser.setLineWrapMode(QTextBrowser.LineWrapMode.WidgetWidth)
        browser.setStyleSheet("QTextBrowser { font-size: 15px; line-height: 1.7; padding: 16px; }")
        browser.setHtml(html_text)
        layout.addWidget(browser)

        button_row = QHBoxLayout()
        button_row.addStretch()
        close_btn = QPushButton("닫기")
        close_btn.setFixedWidth(90)
        close_btn.clicked.connect(dialog.accept)
        button_row.addWidget(close_btn)
        layout.addLayout(button_row)

        dialog.exec()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
