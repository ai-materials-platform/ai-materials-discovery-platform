import os
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
    QStackedWidget,
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

APP_FONT_FAMILY = '"Malgun Gothic"'
APP_FONT_SIZE = 11

LIGHT_QSS = f"""
QMainWindow, QWidget {{ background-color: #F4F6F8; color: #111827; font-family: {APP_FONT_FAMILY}; font-size: {APP_FONT_SIZE}px; }}
QTabWidget::pane {{ border: 1px solid #C9D2DC; background: #FFFFFF; }}
QTabBar {{ background: #E9EEF3; }}
QTabBar::tab {{ background: #E9EEF3; color: #475569; padding: 9px 16px; border: 1px solid #C9D2DC; border-bottom: none; border-top-left-radius: 6px; border-top-right-radius: 6px; }}
QTabBar::tab:selected {{ background: #FFFFFF; color: #111827; font-weight: 700; border-bottom: 2px solid #E56020; }}
QTabBar::tab:hover {{ background: #FFFFFF; color: #111827; }}
QGroupBox {{ border: 1px solid #C9D2DC; border-radius: 10px; margin-top: 12px; padding-top: 12px; font-weight: 700; color: #28323C; background: #FFFFFF; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #475569; letter-spacing: 0.3px; }}
QPushButton {{ background-color: #EEF2F6; color: #111827; border: 1px solid #C9D2DC; border-radius: 8px; padding: 7px 14px; }}
QPushButton:hover {{ background-color: #E2E8F0; }}
QPushButton:disabled {{ color: #7C8794; background: #F4F5F6; border-color: #E2E7EC; }}
QComboBox {{ border: 1px solid #C9D2DC; border-radius: 8px; background: #FFFFFF; padding: 6px 10px; color: #111827; }}
QComboBox:focus {{ border-color: #E56020; }}
QComboBox QAbstractItemView {{ background: #FFFFFF; border: 1px solid #C9D2DC; color: #111827; selection-background-color: #E56020; selection-color: #FFFFFF; }}
QTableWidget {{ border: 1px solid #C9D2DC; gridline-color: #E2E8F0; background: #FFFFFF; alternate-background-color: #F8FAFC; color: #111827; }}
QTableWidget::item {{ background-color: #FFFFFF; color: #111827; }}
QTableWidget::item:alternate {{ background-color: #F8FAFC; }}
QTableWidget::item:selected {{ background: #E56020; color: #FFFFFF; }}
QHeaderView::section {{ background: #E9EEF3; color: #475569; border: none; border-right: 1px solid #C9D2DC; border-bottom: 1px solid #C9D2DC; padding: 7px 10px; font-weight: 700; }}
QScrollBar:vertical {{ width: 8px; background: transparent; }}
QScrollBar::handle:vertical {{ background: #B8C2CE; border-radius: 4px; min-height: 24px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ height: 8px; background: transparent; }}
QScrollBar::handle:horizontal {{ background: #B8C2CE; border-radius: 4px; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
QLineEdit {{ border: 1px solid #C9D2DC; border-radius: 8px; background: #FFFFFF; padding: 7px 10px; color: #111827; selection-background-color: #E56020; }}
QLineEdit:focus {{ border-color: #E56020; }}
QDoubleSpinBox, QSpinBox {{ border: 1px solid #C9D2DC; border-radius: 8px; background: #FFFFFF; padding: 7px 10px; color: #111827; }}
QSplitter::handle {{ background: #C9D2DC; }}
QDialog {{ background: #FFFFFF; }}
QMessageBox {{ background: #FFFFFF; }}
"""

DARK_QSS = f"""
QMainWindow, QWidget {{ background-color: #25282D; color: #F3F4F6; font-family: {APP_FONT_FAMILY}; font-size: {APP_FONT_SIZE}px; }}
QTabWidget {{ background: #25282D; }}
QTabWidget::pane {{ border: 1px solid #4F5965; background: #2F3339; }}
QTabBar {{ background: #25282D; }}
QTabBar::scroller {{ background: #25282D; }}
QTabBar QToolButton {{ background: #25282D; border: none; }}
QTabBar::tab {{ background: #25282D; color: #B6C0CB; padding: 9px 16px; border: 1px solid #4F5965; border-bottom: none; border-top-left-radius: 6px; border-top-right-radius: 6px; }}
QTabBar::tab:selected {{ background: #2F3339; color: #F8FAFC; font-weight: 700; border-bottom: 2px solid #E56020; }}
QTabBar::tab:hover {{ background: #333840; color: #F3F4F6; }}
QGroupBox {{ border: 1px solid #4F5965; border-radius: 10px; margin-top: 12px; padding-top: 12px; font-weight: 700; color: #E2E8F0; background: #2F3339; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #D5DBE3; letter-spacing: 0.3px; }}
QPushButton {{ background-color: #3A4048; color: #F3F4F6; border: 1px solid #59616C; border-radius: 8px; padding: 7px 14px; }}
QPushButton:hover {{ background-color: #454C55; color: #FFFFFF; }}
QPushButton:disabled {{ color: #7D8794; background: #2F3339; border-color: #3F454D; }}
QComboBox {{ border: 1px solid #59616C; border-radius: 8px; background: #2F3339; padding: 6px 10px; color: #F3F4F6; }}
QComboBox:focus {{ border-color: #E56020; }}
QComboBox QAbstractItemView {{ background: #2F3339; border: 1px solid #59616C; color: #F3F4F6; selection-background-color: #E56020; selection-color: #FFFFFF; }}
QTableWidget {{ border: 1px solid #4F5965; gridline-color: #3A4048; background: #2F3339; alternate-background-color: #25282D; color: #F3F4F6; }}
QTableWidget::item {{ background-color: #2F3339; color: #F3F4F6; }}
QTableWidget::item:alternate {{ background-color: #25282D; }}
QTableWidget::item:selected {{ background: #E56020; color: #FFFFFF; }}
QHeaderView::section {{ background: #25282D; color: #D5DBE3; border: none; border-right: 1px solid #4F5965; border-bottom: 1px solid #4F5965; padding: 7px 10px; font-weight: 700; }}
QScrollBar:vertical {{ width: 8px; background: transparent; }}
QScrollBar::handle:vertical {{ background: #59616C; border-radius: 4px; min-height: 24px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ height: 8px; background: transparent; }}
QScrollBar::handle:horizontal {{ background: #59616C; border-radius: 4px; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
QLineEdit {{ border: 1px solid #59616C; border-radius: 8px; background: #2F3339; padding: 7px 10px; color: #F3F4F6; selection-background-color: #E56020; }}
QLineEdit:focus {{ border-color: #E56020; }}
QDoubleSpinBox, QSpinBox {{ border: 1px solid #59616C; border-radius: 8px; background: #2F3339; padding: 7px 10px; color: #F3F4F6; }}
QSplitter::handle {{ background: #4F5965; }}
QDialog {{ background: #2F3339; color: #F3F4F6; }}
QMessageBox {{ background: #2F3339; color: #F3F4F6; }}
QScrollArea {{ background: transparent; border: none; }}
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
        self.pretrained_model_engine = None
        self.pretrained_data_engine = None
        self.pretrained_model_type = None
        self.pretrained_metrics = None
        self.preprocessing_ready = False
        self._open_dialogs = []
        self.last_r2_avg = None
        self._ui_font_family = "Malgun Gothic"
        self._ui_font_size = APP_FONT_SIZE
        self._dark_mode = False
        self._panel_widgets = []       # background: panel
        self._panel_header_widgets = []
        self._divider_widgets = []     # background: divider
        self._info_box_widgets = []    # info box (accent left border)
        self._section_lbl_widgets = [] # color: text_label, monospace
        self._muted_bg_widgets = []    # background: summary/muted
        self._prediction_input_groups = []
        self._prediction_input_fields = []
        self._prediction_input_labels = []

        self.init_ui()

    def init_ui(self):
        self.setStyleSheet(GLOBAL_QSS)
        self._apply_ui_font()

        # Menu bar
        mb = self.menuBar()
        mb.setStyleSheet(
            "QMenuBar { background: #252525; color: #D7DCE3; font-size: 12px; padding: 3px 6px; border-bottom: 1px solid #363636; }"
            "QMenuBar::item { background: transparent; padding: 5px 10px; }"
            "QMenuBar::item:selected { background: #3A4048; color: #FFFFFF; }"
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

        self.main_mode_stack = QStackedWidget()
        root_layout.addWidget(self.main_mode_stack, 1)

        self.material_prediction_page = self._create_material_prediction_page()
        self.main_mode_stack.addWidget(self.material_prediction_page)

        user_page = QWidget()
        user_layout = QVBoxLayout(user_page)
        user_layout.setContentsMargins(0, 0, 0, 0)
        user_layout.setSpacing(0)

        # 3-panel splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(2)

        splitter.addWidget(self._create_left_panel())
        splitter.addWidget(self._create_settings_panel())

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        splitter.addWidget(self.tabs)
        splitter.setSizes([190, 270, 860])

        user_layout.addWidget(splitter, 1)
        self.main_mode_stack.addWidget(user_page)
        self._switch_main_mode(0)

        # Status bar
        sb = self.statusBar()
        sb.setStyleSheet(
            f"QStatusBar {{ background: #252525; color: white; font-size: 11px; "
            f"padding: 0 8px; border-top: 1px solid #363636; }}"
            "QStatusBar::item { border: none; }"
        )
        self._sb_status = QLabel("● 준비완료")
        self._sb_samples = QLabel("샘플: —")
        self._sb_missing = QLabel("결측: —")
        self._sb_model = QLabel("모델: 미훈련")
        _lbl_style = "color: #F8FAFC; font-size: 11px; font-weight: 600; padding: 0 10px;"
        for lbl in [self._sb_samples, self._sb_missing, self._sb_model]:
            lbl.setStyleSheet(_lbl_style)
        self._sb_status.setStyleSheet("color: #58C472; font-size: 11px; font-weight: 700; padding: 0 10px;")
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
        self._apply_theme_colors()
        self.prepare_pretrained_model()

    def _apply_ui_font(self):
        font = QFont(self._ui_font_family)
        font.setPixelSize(self._ui_font_size)
        font.setStyleHint(QFont.StyleHint.SansSerif)
        QApplication.instance().setFont(font)
        self.setFont(font)
        plt.rcParams["font.family"] = ["Malgun Gothic", "sans-serif"]
        if hasattr(self, "canvas"):
            self.render_training_placeholder()
        if hasattr(self, "perf_canvas"):
            self.render_performance_placeholder()

    # ── Toolbar ──────────────────────────────────────────────────────────────

    def _create_toolbar(self):
        bar = QWidget()
        self._toolbar_widget = bar
        bar.setFixedHeight(48)
        bar.setStyleSheet("background: #FFFFFF; border-bottom: 1px solid #C9D2DC;")

        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 0, 14, 0)
        layout.setSpacing(12)

        self._mode_nav_widget = QWidget()
        mode_layout = QHBoxLayout(self._mode_nav_widget)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(8)

        self.material_mode_btn = QPushButton("물성예측")
        self.material_mode_btn.setCheckable(True)
        self.material_mode_btn.clicked.connect(lambda: self._switch_main_mode(0))
        mode_layout.addWidget(self.material_mode_btn)

        self.user_mode_btn = QPushButton("User")
        self.user_mode_btn.setCheckable(True)
        self.user_mode_btn.clicked.connect(lambda: self._switch_main_mode(1))
        mode_layout.addWidget(self.user_mode_btn)

        layout.addWidget(self._mode_nav_widget)
        layout.addStretch()

        self._toolbar_title = QLabel("AI Materials Discovery Platform")
        self._toolbar_title.setStyleSheet(
            "color: #111827; font-size: 15px; font-weight: 700; letter-spacing: 0.2px;"
        )
        layout.addWidget(self._toolbar_title)

        self._theme_btn = QPushButton("다크 모드")
        self._theme_btn.setFixedHeight(32)
        self._theme_btn.setMinimumWidth(92)
        self._theme_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._theme_btn.setStyleSheet(
            "QPushButton { background: #111827; color: #F8FAFC; border: 1px solid #111827; "
            "border-radius: 16px; font-size: 11px; font-weight: 700; padding: 0 14px; }"
            "QPushButton:hover { background: #1F2937; border-color: #334155; }"
        )
        self._theme_btn.clicked.connect(self._toggle_theme)
        layout.addWidget(self._theme_btn)
        self._update_mode_buttons()
        return bar

    def _switch_main_mode(self, index):
        if hasattr(self, "main_mode_stack"):
            self.main_mode_stack.setCurrentIndex(index)
        self._current_main_mode = index
        self._update_mode_buttons()

    def _update_mode_buttons(self):
        if not hasattr(self, "material_mode_btn") or not hasattr(self, "user_mode_btn"):
            return
        current_index = getattr(self, "_current_main_mode", 0)
        self.material_mode_btn.setChecked(current_index == 0)
        self.user_mode_btn.setChecked(current_index == 1)
        self._apply_mode_button_styles()

    def _apply_mode_button_styles(self):
        active_bg = "#E56020"
        active_text = "#FFFFFF"
        if self._dark_mode:
            inactive_bg = "#2F3339"
            inactive_text = "#D5DBE3"
            inactive_border = "#4F5965"
            hover_bg = "#3A4048"
        else:
            inactive_bg = "#FFFFFF"
            inactive_text = "#475569"
            inactive_border = "#C9D2DC"
            hover_bg = "#EEF2F6"

        button_style = (
            "QPushButton { "
            f"background: {inactive_bg}; color: {inactive_text}; border: 1px solid {inactive_border}; "
            "border-radius: 16px; font-size: 11px; font-weight: 700; padding: 0 16px; min-height: 32px; }"
            f"QPushButton:hover {{ background: {hover_bg}; }}"
            f"QPushButton:checked {{ background: {active_bg}; color: {active_text}; border: 1px solid {active_bg}; }}"
        )
        self.material_mode_btn.setStyleSheet(button_style)
        self.user_mode_btn.setStyleSheet(button_style)

    def _create_material_prediction_page(self):
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(24, 24, 24, 24)
        outer.setSpacing(18)

        content_row = QHBoxLayout()
        content_row.setSpacing(18)

        input_card = QGroupBox("사전학습 모델 입력")
        input_layout = QVBoxLayout(input_card)
        self.pretrained_active_model_info = QLabel("사용 중인 모델: 사전학습 모델 없음")
        self.pretrained_active_model_info.hide()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        form_container = QWidget()
        form_layout = QVBoxLayout(form_container)
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.setSpacing(12)

        self.pretrained_inputs = {}
        self._build_prediction_input_sections(form_layout, self.pretrained_inputs)
        scroll.setWidget(form_container)
        input_layout.addWidget(scroll, 1)

        self.pretrained_predict_btn = QPushButton("사전학습 모델로 예측")
        self.pretrained_predict_btn.setFixedHeight(42)
        self.pretrained_predict_btn.setStyleSheet(
            "QPushButton { background: #16A34A; color: white; border: none; border-radius: 10px; font-weight: 700; }"
            "QPushButton:hover { background: #15803D; }"
        )
        self.pretrained_predict_btn.clicked.connect(self.on_pretrained_predict_clicked)
        input_layout.addWidget(self.pretrained_predict_btn)
        content_row.addWidget(input_card, 1)

        result_card = QGroupBox("예측 결과")
        result_layout = QVBoxLayout(result_card)
        self.pretrained_result_tabs = QTabWidget()

        result_tab = QWidget()
        result_tab_layout = QVBoxLayout(result_tab)
        self.pretrained_result_display = QLabel(
            "<b>예측 준비 완료</b><br>사전학습 모델로 바로 물성 예측을 실행할 수 있습니다."
        )
        self.pretrained_result_display.setWordWrap(True)
        result_tab_layout.addWidget(self.pretrained_result_display)
        self.pretrained_prediction_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        result_tab_layout.addWidget(self.pretrained_prediction_canvas)
        self.pretrained_result_tabs.addTab(result_tab, "예측 결과")

        curve_tab = QWidget()
        curve_layout = QVBoxLayout(curve_tab)
        self.pretrained_curve_placeholder = QLabel(
            "Stress-Strain Curve 탭은 현재 자리만 준비되어 있습니다."
        )
        self.pretrained_curve_placeholder.setWordWrap(True)
        curve_layout.addWidget(self.pretrained_curve_placeholder)
        self.pretrained_result_tabs.addTab(curve_tab, "Stress-Strain Curve")

        result_layout.addWidget(self.pretrained_result_tabs)

        self.pretrained_status_label = QLabel("")
        self.pretrained_status_label.setWordWrap(True)
        self.pretrained_status_label.hide()

        self.pretrained_model_summary_label = QLabel("")
        self.pretrained_model_summary_label.setWordWrap(True)
        self.pretrained_model_summary_label.hide()
        content_row.addWidget(result_card, 1)

        outer.addLayout(content_row, 1)
        return page

    def _build_prediction_input_sections(self, parent_layout, input_store):
        comp_group = QGroupBox("합금 조성 (wt%)")
        comp_group_layout = QVBoxLayout(comp_group)
        comp_group_layout.setContentsMargins(12, 12, 12, 12)
        comp_group_layout.setSpacing(12)
        composition_defaults = {
            "Fe": "96.0",
            "C": "0.08",
            "Si": "0.4",
            "Mn": "1.5",
            "P": "0.01",
            "S": "0.005",
            "Ni": "0.2",
            "Cr": "0.3",
            "Mo": "0.05",
            "Cu": "0.1",
            "V": "0.01",
            "N": "0.005",
            "Nb": "0.02",
            "Ti": "0.01",
            "B": "0.0005",
            "Al": "0.03",
        }
        composition_items = list(composition_defaults.items())
        midpoint = (len(composition_items) + 1) // 2
        comp_columns = QHBoxLayout()
        comp_columns.setContentsMargins(0, 0, 0, 0)
        comp_columns.setSpacing(16)
        comp_columns.addLayout(self._create_prediction_form(composition_items[:midpoint], input_store))
        comp_columns.addLayout(self._create_prediction_form(composition_items[midpoint:], input_store))
        comp_group_layout.addLayout(comp_columns)
        parent_layout.addWidget(comp_group)
        self._prediction_input_groups.append(comp_group)

        proc_group = QGroupBox("공정 및 조직")
        proc_group_layout = QVBoxLayout(proc_group)
        proc_group_layout.setContentsMargins(12, 12, 12, 12)
        proc_group_layout.setSpacing(12)
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
        proc_group_layout.addLayout(self._create_prediction_form(list(proc_defaults.items()), input_store))
        parent_layout.addWidget(proc_group)
        self._prediction_input_groups.append(proc_group)
        self._apply_prediction_input_styles()

    def _create_prediction_form(self, items, input_store):
        form_layout = QFormLayout()
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.setHorizontalSpacing(14)
        form_layout.setVerticalSpacing(10)
        form_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        for col, value in items:
            label = QLabel(col)
            line_edit = QLineEdit()
            line_edit.setText(value)
            form_layout.addRow(label, line_edit)
            input_store[col] = line_edit
            self._prediction_input_labels.append(label)
            self._prediction_input_fields.append(line_edit)
        return form_layout

    def _apply_prediction_input_styles(self):
        c = self._theme()
        group_style = (
            "QGroupBox { "
            f"background: {c['panel_bg']}; border: 1px solid {c['border']}; border-radius: 12px; "
            "margin-top: 10px; padding-top: 14px; font-weight: 700; }"
            f"QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 6px; color: {c['text_primary']}; }}"
        )
        line_edit_style = (
            "QLineEdit { "
            f"background: {c['input_bg']}; color: {c['text_primary']}; border: 1px solid {c['border']}; "
            "border-radius: 8px; padding: 7px 10px; selection-background-color: #E56020; selection-color: #FFFFFF; }"
            "QLineEdit:focus { border-color: #E56020; }"
        )
        for group in self._prediction_input_groups:
            group.setStyleSheet(group_style)
        for label in self._prediction_input_labels:
            label.setStyleSheet(
                f"color: {c['text_sec']}; font-size: 12px; font-weight: 600; padding-right: 4px;"
            )
        for field in self._prediction_input_fields:
            field.setStyleSheet(line_edit_style)

    def _toggle_theme(self):
        self._dark_mode = not self._dark_mode
        self._theme_btn.setText("라이트 모드" if self._dark_mode else "다크 모드")
        self.setStyleSheet(DARK_QSS if self._dark_mode else LIGHT_QSS)
        self._apply_theme_colors()
        if hasattr(self, "canvas") and self.last_r2_avg is None:
            self.render_training_placeholder()
        if hasattr(self, "perf_canvas") and self.last_r2_avg is None:
            self.render_performance_placeholder()

    def _theme(self):
        d = self._dark_mode
        return {
            "app_bg":      "#25282D" if d else "#F4F6F8",
            "panel_bg":    "#2F3339" if d else "#FFFFFF",
            "muted_bg":    "#343941" if d else "#F8FAFC",
            "info_bg":     "#3A4048" if d else "#F6F8FB",
            "divider":     "#4F5965" if d else "#CBD5E1",
            "border":      "#4F5965" if d else "#C9D2DC",
            "text_primary":"#F3F4F6" if d else "#111827",
            "text_sec":    "#D5DBE3" if d else "#334155",
            "text_label":  "#B6C0CB" if d else "#64748B",
            "tb_bg":       "#1E1E1E" if d else "#FFFFFF",
            "tb_border":   "#3B4350" if d else "#C9D2DC",
            "status_bg":   "#1E1E1E" if d else "#FFFFFF",
            "status_text": "#F8FAFC" if d else "#111827",
            "status_muted":"#D7DCE3" if d else "#334155",
            "input_bg":    "#2F3339" if d else "#FFFFFF",
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
        if hasattr(self, "_toolbar_title"):
            self._toolbar_title.setStyleSheet(
                f"color: {'#F8FAFC' if self._dark_mode else '#111827'}; font-size: 15px; font-weight: 700; letter-spacing: 0.2px;"
            )
        if hasattr(self, "_theme_btn"):
            self._theme_btn.setText("라이트 모드" if self._dark_mode else "다크 모드")
            theme_bg = "#FFFFFF" if self._dark_mode else "#111827"
            theme_text = "#111827" if self._dark_mode else "#F8FAFC"
            theme_border = c["border"] if self._dark_mode else "#111827"
            theme_hover = "#F8FAFC" if self._dark_mode else "#1F2937"
            theme_hover_border = "#CBD5E1" if self._dark_mode else "#334155"
            self._theme_btn.setStyleSheet(
                "QPushButton { "
                f"background: {theme_bg}; color: {theme_text}; border: 1px solid {theme_border}; "
                "border-radius: 16px; font-size: 11px; font-weight: 700; padding: 0 14px; }"
                f"QPushButton:hover {{ background: {theme_hover}; border-color: {theme_hover_border}; }}"
            )
        menu_bg = c["tb_bg"] if self._dark_mode else "#FFFFFF"
        menu_text = c["text_sec"] if self._dark_mode else "#111827"
        menu_border = c["tb_border"] if self._dark_mode else c["border"]
        menu_hover_bg = c["tb_border"] if self._dark_mode else "#EEF2F6"
        menu_hover_text = "#FFFFFF" if self._dark_mode else "#111827"
        self.menuBar().setStyleSheet(
            f"QMenuBar {{ background: {menu_bg}; color: {menu_text}; font-size: 12px; padding: 3px 6px; border-bottom: 1px solid {menu_border}; }}"
            f"QMenuBar::item {{ background: transparent; padding: 5px 10px; color: {menu_text}; }}"
            f"QMenuBar::item:selected {{ background: {menu_hover_bg}; color: {menu_hover_text}; }}"
        )
        self.statusBar().setStyleSheet(
            f"QStatusBar {{ background: {c['status_bg']}; color: {c['status_text']}; font-size: 11px; "
            f"padding: 0 8px; border-top: 1px solid {c['border']}; }}"
            "QStatusBar::item { border: none; }"
        )

        # ── Panel backgrounds ──
        for w in self._panel_widgets:
            w.setStyleSheet(f"background: {c['panel_bg']};")

        for w in self._panel_header_widgets:
            w.setStyleSheet(
                f"font-size: 11px; color: {c['text_label']}; padding: 10px 14px; "
                f"letter-spacing: 0.8px; font-weight: 600; border-bottom: 1px solid {c['divider']};"
            )

        # ── Dividers ──
        for w in self._divider_widgets:
            w.setStyleSheet(f"background: {c['divider']};")

        # ── Info boxes (accent left border) ──
        for w in self._info_box_widgets:
            w.setStyleSheet(
                f"font-size: 12px; color: {c['text_sec']}; background: {c['info_bg']}; "
                f"padding: 10px; border-left: 3px solid {c['accent']}; border-radius: 6px;"
            )

        # ── Section labels (monospace, label color) ──
        for w in self._section_lbl_widgets:
            w.setStyleSheet(
                f"font-size: 11px; color: {c['text_label']}; "
                "font-weight: 600; letter-spacing: 0.4px;"
            )

        # ── Muted background boxes (summary etc.) ──
        for w in self._muted_bg_widgets:
            w.setStyleSheet(
                f"font-size: 12px; color: {c['text_sec']}; padding: 10px; "
                f"background: {c['muted_bg']}; border: 1px solid {c['border']}; border-radius: 6px;"
            )

        # ── Settings misc labels ──
        self.file_path_label.setStyleSheet(f"font-size: 12px; color: {c['text_sec']};")
        self.status_label.setStyleSheet(f"color: {c['text_sec']}; font-size: 11px;")
        self.domain_range_status_label.setStyleSheet(f"color: {c['text_sec']}; font-size: 11px;")
        self.reset_preprocess_btn.setStyleSheet(
            f"background: {c['muted_bg']}; color: {c['text_sec']}; border: 1px solid {c['border']}; border-radius: 6px; font-size: 12px; font-weight: 600;"
        )

        self._sb_status.setStyleSheet(
            f"color: {c['accent'] if self.preprocessing_ready and not self.model_engine else '#58C472'}; "
            "font-size: 11px; font-weight: 700; padding: 0 10px;"
        )
        for lbl in [self._sb_samples, self._sb_missing, self._sb_model]:
            lbl.setStyleSheet(
                f"color: {c['status_muted']}; font-size: 11px; font-weight: 600; padding: 0 10px;"
            )

        # ── Tree section title labels ──
        for w in self._tree_section_title_lbls:
            w.setStyleSheet(
                f"font-size: 11px; color: {c['text_label']}; font-weight: 700; letter-spacing: 0.4px;"
            )

        info_bg = "#15324A" if self._dark_mode else "#EFF6FF"
        info_border = "#2563EB" if self._dark_mode else "#BFDBFE"
        info_text = "#DBEAFE" if self._dark_mode else "#1D4ED8"
        warn_bg = "#4A3818" if self._dark_mode else "#FFF7D6"
        warn_border = "#D4A72C" if self._dark_mode else "#FACC15"
        warn_text = "#FDE68A" if self._dark_mode else "#7A5D00"
        success_bg = "#173B2F" if self._dark_mode else "#ECFDF3"
        success_border = "#22C55E" if self._dark_mode else "#86EFAC"
        success_text = "#DCFCE7" if self._dark_mode else "#166534"

        if hasattr(self, "processed_preview_info_label"):
            self.processed_preview_info_label.setStyleSheet(
                f"font-size: 11px; color: {c['text_sec']}; padding: 8px 10px; "
                f"background: {c['muted_bg']}; border-top: 1px solid {c['divider']};"
            )
        if self._prediction_input_groups or self._prediction_input_fields or self._prediction_input_labels:
            self._apply_prediction_input_styles()
        if hasattr(self, "feature_selection_status_label"):
            self.feature_selection_status_label.setStyleSheet(
                f"background-color: {warn_bg}; padding: 10px; border-radius: 8px; "
                f"color: {warn_text}; border: 1px solid {warn_border}; font-weight: 600;"
            )
        if hasattr(self, "training_data_status_label"):
            self.training_data_status_label.setStyleSheet(
                f"background-color: {warn_bg}; padding: 10px; border-radius: 8px; "
                f"color: {warn_text}; border: 1px solid {warn_border}; font-weight: 600;"
            )
        if hasattr(self, "active_model_info"):
            card_bg = success_bg if self.model_engine else info_bg
            card_border = success_border if self.model_engine else info_border
            card_text = success_text if self.model_engine else info_text
            self.active_model_info.setStyleSheet(
                f"background-color: {card_bg}; color: {card_text}; padding: 11px 12px; "
                f"border: 1px solid {card_border}; border-radius: 10px; font-weight: 700; margin-bottom: 12px;"
            )
        if hasattr(self, "result_display"):
            self.result_display.setStyleSheet(
                f"font-size: 12px; font-weight: 600; color: {c['text_primary']}; "
                f"background: {c['muted_bg']}; border: 1px solid {c['border']}; "
                "border-radius: 10px; padding: 12px;"
            )
        if hasattr(self, "comp_group_title_label"):
            self.comp_group_title_label.setStyleSheet(
                f"font-size: 14px; font-weight: 700; color: {c['text_primary']}; padding: 0 2px 4px 2px;"
            )
        if hasattr(self, "proc_group_title_label"):
            self.proc_group_title_label.setStyleSheet(
                f"font-size: 14px; font-weight: 700; color: {c['text_primary']}; padding: 0 2px 4px 2px;"
            )
        if hasattr(self, "inference_tab"):
            self.inference_tab.setStyleSheet(f"background: {c['app_bg']};")
        if hasattr(self, "inference_left_frame"):
            self.inference_left_frame.setStyleSheet(
                f"background: {c['panel_bg']}; border: 1px solid {c['border']}; border-radius: 12px;"
            )
        if hasattr(self, "inference_right_frame"):
            self.inference_right_frame.setStyleSheet(
                f"background: {c['panel_bg']}; border: 1px solid {c['border']}; border-radius: 12px;"
            )
        if hasattr(self, "ws_detail_widget"):
            self.ws_detail_widget.setStyleSheet(
                f"background-color: {c['muted_bg']}; border-top: 1px solid {c['divider']};"
            )
        if hasattr(self, "ws_detail_thumb"):
            self.ws_detail_thumb.setStyleSheet(
                f"border: 1px solid {c['border']}; background: {c['panel_bg']}; border-radius: 8px; color: {c['text_label']};"
            )
        if hasattr(self, "ws_detail_perf_thumb"):
            self.ws_detail_perf_thumb.setStyleSheet(
                f"border: 1px solid {c['border']}; background: {c['panel_bg']}; border-radius: 8px; color: {c['text_label']};"
            )
        if hasattr(self, "ws_detail_info"):
            self.ws_detail_info.setStyleSheet(
                f"font-size: 12px; color: {c['text_sec']}; padding-left: 12px;"
            )
        if hasattr(self, "ws_hint_label"):
            self.ws_hint_label.setStyleSheet(
                f"color: {c['text_label']}; font-size: 11px; font-weight: 600; margin-top: 4px;"
            )
        if hasattr(self, "perf_header_label"):
            self.perf_header_label.setStyleSheet(
                f"font-size: 16px; font-weight: 700; color: {c['text_primary']};"
            )
        if hasattr(self, "perf_desc_label"):
            self.perf_desc_label.setStyleSheet(
                f"color: {c['text_sec']}; font-weight: 600;"
            )
        if hasattr(self, "ws_name_label"):
            self.ws_name_label.setStyleSheet(
                f"font-size: 12px; color: {c['text_sec']}; font-weight: 600;"
            )
        if hasattr(self, "ws_list_title"):
            self.ws_list_title.setStyleSheet(
                f"font-size: 14px; font-weight: 700; color: {c['text_primary']};"
            )
        if hasattr(self, "ws_train_thumb_label"):
            self.ws_train_thumb_label.setStyleSheet(
                f"font-size: 11px; color: {c['text_sec']}; font-weight: 600;"
            )
        if hasattr(self, "ws_perf_thumb_label"):
            self.ws_perf_thumb_label.setStyleSheet(
                f"font-size: 11px; color: {c['text_sec']}; font-weight: 600;"
            )
        if hasattr(self, "ws_compare_btn"):
            self.ws_compare_btn.setStyleSheet(
                "QPushButton { "
                f"background: {c['accent']}; color: white; border: none; border-radius: 17px; "
                "font-weight: 700; padding: 6px 12px; }"
                "QPushButton:hover { background: #F97316; }"
            )
        if hasattr(self, "ws_refresh_btn"):
            self.ws_refresh_btn.setStyleSheet(
                "QPushButton { "
                f"background: {c['panel_bg']}; color: {c['text_sec']}; border: 1px solid {c['border']}; "
                "border-radius: 17px; font-weight: 700; padding: 6px 12px; }"
                f"QPushButton:hover {{ background: {c['muted_bg']}; border-color: {c['divider']}; }}"
            )
        if hasattr(self, "ws_save_btn"):
            self.ws_save_btn.setStyleSheet(
                "QPushButton { background: #E56020; color: white; border: none; border-radius: 10px; "
                "font-weight: 700; padding: 7px 16px; }"
                "QPushButton:hover { background: #F97316; }"
            )
        if hasattr(self, "ws_load_btn"):
            self.ws_load_btn.setStyleSheet(
                "QPushButton { "
                f"background: {c['panel_bg']}; color: {c['text_sec']}; border: 1px solid {c['border']}; "
                "border-radius: 10px; font-weight: 700; padding: 7px 14px; }"
                f"QPushButton:hover {{ background: {c['muted_bg']}; border-color: {c['divider']}; }}"
            )
        if hasattr(self, "ws_delete_btn"):
            self.ws_delete_btn.setStyleSheet(
                "QPushButton { background: #DC2626; color: white; border: none; border-radius: 10px; "
                "font-weight: 700; padding: 7px 12px; }"
                "QPushButton:hover { background: #EF4444; }"
            )
        if hasattr(self, "ws_panel_load_btn"):
            self.ws_panel_load_btn.setStyleSheet(
                "QPushButton { background: #0F766E; color: white; border: none; border-radius: 10px; "
                "font-weight: 700; padding: 8px 14px; }"
                "QPushButton:hover { background: #0D9488; }"
            )
        if hasattr(self, "preprocessing_tab"):
            self.preprocessing_tab.setStyleSheet(f"background: {c['panel_bg']};")
        if hasattr(self, "workspace_tab"):
            self.workspace_tab.setStyleSheet(f"background: {c['panel_bg']};")
        if hasattr(self, "_settings_header_row"):
            self._settings_header_row.setStyleSheet(
                f"background: {c['panel_bg']}; border-bottom: 1px solid {c['divider']};"
            )
        if hasattr(self, "_settings_header_label"):
            self._settings_header_label.setStyleSheet(
                f"font-size: 11px; color: {c['text_label']}; padding: 10px 14px; "
                "letter-spacing: 0.8px; font-weight: 600;"
            )
        if hasattr(self, "_settings_help_btn"):
            self._settings_help_btn.setStyleSheet(
                "QPushButton { "
                f"background: transparent; color: {c['text_label']}; border: 1px solid {c['border']}; "
                "border-radius: 10px; font-size: 11px; font-weight: 700; padding: 0; margin: 4px 8px 4px 0; }"
                f"QPushButton:hover {{ background: {c['muted_bg']}; color: {c['text_primary']}; }}"
            )

        if hasattr(self, "ws_name_input"):
            self.ws_name_input.setStyleSheet(
                f"QLineEdit {{ background: {c['input_bg']}; color: {c['text_primary']}; "
                f"border: 1px solid {c['border']}; border-radius: 6px; padding: 6px 10px; }}"
            )
        if hasattr(self, "ws_combo"):
            self.ws_combo.setStyleSheet(
                f"QComboBox {{ background: {c['input_bg']}; color: {c['text_primary']}; "
                f"border: 1px solid {c['border']}; border-radius: 6px; padding: 5px 10px; }}"
            )
        if hasattr(self, "_tree_file_label"):
            self._update_project_tree()

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
            "font-size: 11px; color: #5B6470; padding: 10px 14px; letter-spacing: 0.8px; "
            "font-weight: 600; border-bottom: 1px solid #E1E5EA;"
        )
        self._panel_widgets.append(header)
        self._panel_header_widgets.append(header)
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
            t.setStyleSheet("font-size: 11px; color: #5B6470; font-weight: 700; letter-spacing: 0.4px;")
            self._tree_section_title_lbls.append(t)
            value_lbl.setStyleSheet("font-size: 12px; color: #111827; padding-left: 6px;")
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
        colors = self._theme()
        primary_color = colors["text_sec"]
        muted_color = colors["text_label"]
        accent_color = colors["accent"]

        if self.data_engine.file_path:
            self._tree_file_label.setText(os.path.basename(self.data_engine.file_path))
            self._tree_file_label.setStyleSheet(f"font-size: 12px; color: {primary_color}; padding-left: 6px; font-weight: 600;")
        else:
            self._tree_file_label.setText("(없음)")
            self._tree_file_label.setStyleSheet(f"font-size: 12px; color: {muted_color}; padding-left: 6px;")

        if self.preprocessing_ready:
            n = self.data_engine.last_quality_report.get("rows_after", "?")
            self._tree_preprocess_label.setText(f"완료 ({n}행)")
            self._tree_preprocess_label.setStyleSheet(f"font-size: 12px; color: {accent_color}; padding-left: 6px; font-weight: 700;")
        else:
            self._tree_preprocess_label.setText("미실행")
            self._tree_preprocess_label.setStyleSheet(f"font-size: 12px; color: {muted_color}; padding-left: 6px;")

        if self.model_engine:
            nm = {"RF": "Random Forest", "GBM": "Gradient Boosting", "MLP": "Neural Network", "TFP": "TFP"}
            self._tree_model_label.setText(nm.get(self.model_type, self.model_type))
            self._tree_model_label.setStyleSheet(f"font-size: 12px; color: {accent_color}; padding-left: 6px; font-weight: 700;")
        else:
            self._tree_model_label.setText("미학습")
            self._tree_model_label.setStyleSheet(f"font-size: 12px; color: {muted_color}; padding-left: 6px;")

        if self.last_r2_avg is not None:
            self._tree_results_label.setText(f"R² {self.last_r2_avg * 100:.1f}%")
            self._tree_results_label.setStyleSheet(f"font-size: 12px; color: {accent_color}; padding-left: 6px; font-weight: 700;")
        else:
            self._tree_results_label.setText("—")
            self._tree_results_label.setStyleSheet(f"font-size: 12px; color: {muted_color}; padding-left: 6px;")

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

        self._settings_header_row = QWidget()
        self._panel_widgets.append(self._settings_header_row)
        hdr_row = QHBoxLayout(self._settings_header_row)
        hdr_row.setContentsMargins(0, 0, 0, 0)
        hdr_row.setSpacing(0)

        self._settings_header_label = QLabel("설정")
        self._settings_header_label.setStyleSheet(
            "font-size: 11px; color: #5B6470; padding: 10px 14px; letter-spacing: 0.8px; "
            "font-weight: 600;"
        )
        self._settings_help_btn = QPushButton("?")
        self._settings_help_btn.setFixedSize(20, 20)
        self._settings_help_btn.setStyleSheet(
            "QPushButton { background: transparent; color: #5B6470; border: 1px solid #CAD0D7; "
            "border-radius: 10px; font-size: 11px; font-weight: 700; padding: 0; margin: 4px 8px 4px 0; }"
            "QPushButton:hover { background: #EEF1F4; color: #111827; }"
        )
        self._settings_help_btn.clicked.connect(self.show_quality_help)
        hdr_row.addWidget(self._settings_header_label)
        hdr_row.addStretch()
        hdr_row.addWidget(self._settings_help_btn)
        outer.addWidget(self._settings_header_row)

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
        self.file_path_label.setStyleSheet("font-size: 12px; color: #374151;")
        layout.addWidget(self.file_path_label)
        self.select_file_btn = QPushButton("파일 열기  (.xls / .xlsx)")
        self.select_file_btn.setFixedHeight(34)
        self.select_file_btn.setStyleSheet(
            "background: #E56020; color: white; border: none; border-radius: 6px; font-size: 12px; font-weight: 700;"
        )
        self.select_file_btn.clicked.connect(self.on_select_file_clicked)
        layout.addWidget(self.select_file_btn)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #374151; font-size: 11px;")
        layout.addWidget(self.status_label)

        layout.addWidget(s_divider())

        # 02 도메인 검증
        layout.addWidget(s_label("02 — 도메인 검증"))
        self.domain_rule_label = QLabel(
            "오스테나이트 조성 기준과 고온 특성 기준 두 부류로 범위를 확인합니다."
        )
        self.domain_rule_label.setWordWrap(True)
        self.domain_rule_label.setStyleSheet(
            "font-size: 12px; color: #374151; background: #F4F5F7; padding: 10px; border-left: 3px solid #E56020; border-radius: 6px;"
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
        self.domain_range_status_label.setStyleSheet("color: #374151; font-size: 11px;")
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
            "font-size: 12px; color: #374151; background: #F4F5F7; padding: 10px; border-left: 3px solid #E56020; border-radius: 6px;"
        )
        self._info_box_widgets.append(self.feature_engineering_label)
        layout.addWidget(self.feature_engineering_label)

        self.quality_summary_label = QLabel("전처리 결과 요약이 아직 없습니다.")
        self.quality_summary_label.setWordWrap(True)
        self.quality_summary_label.setStyleSheet(
            "font-size: 12px; color: #374151; padding: 10px; background: #F6F7F9; border: 1px solid #D3D7DC; border-radius: 6px;"
        )
        self._muted_bg_widgets.append(self.quality_summary_label)
        layout.addWidget(self.quality_summary_label)

        layout.addWidget(s_divider())

        # Action buttons
        self.preprocess_btn = QPushButton("전처리 실행")
        self.preprocess_btn.setFixedHeight(38)
        self.preprocess_btn.setStyleSheet(
            "background: #E56020; color: white; border: none; border-radius: 6px; font-size: 13px; font-weight: 700; letter-spacing: 0.5px;"
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
            "background: #F6F7F9; color: #374151; border: 1px solid #D3D7DC; border-radius: 6px; font-size: 12px; font-weight: 600;"
        )
        self.reset_preprocess_btn.clicked.connect(self.on_reset_preprocessing_clicked)
        layout.addWidget(self.reset_preprocess_btn)

        self.go_to_training_btn = QPushButton("→ 학습 컬럼 선택으로")
        self.go_to_training_btn.setFixedHeight(30)
        self.go_to_training_btn.setEnabled(False)
        self.go_to_training_btn.clicked.connect(lambda: self.tabs.setCurrentIndex(1))
        self.go_to_training_btn.hide()

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
        lbl.setStyleSheet("font-size: 11px; color: #5B6470; font-weight: 600; letter-spacing: 0.4px;")
        return lbl

    def _s_divider(self):
        div = QWidget()
        div.setFixedHeight(1)
        div.setStyleSheet("background: #EBEBEA;")
        return div

    def setup_preprocessing_tab(self):
        tab = QWidget()
        self.preprocessing_tab = tab
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
            "font-size: 11px; color: #374151; padding: 8px 10px; background: #F6F7F9; border-top: 1px solid #D3D7DC;"
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
            "background-color: #FFF7D6; padding: 10px; border-radius: 8px; color: #7A5D00; border: 1px solid #FACC15;"
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
        model_help_btn.setFixedHeight(32)
        model_help_btn.setStyleSheet(
            "QPushButton { background: #FFFFFF; color: #334155; border: 1px solid #CBD5E1; "
            "border-radius: 16px; font-weight: 700; padding: 0 14px; }"
            "QPushButton:hover { background: #F8FAFC; border-color: #94A3B8; }"
        )
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
        self.training_input_combo.addItem("데이터 정제 + 합금 지표", "combined")
        self.training_input_combo.setCurrentIndex(0)
        self.training_input_combo.hide()
        info_layout.addWidget(model_selection_group)

        help_label = QLabel(
            "반복 횟수를 크게 늘린다고 항상 성능이 좋아지지는 않습니다.\n기본값으로 먼저 학습한 뒤 필요할 때만 조정하는 것을 권장합니다."
        )
        help_label.setWordWrap(True)
        help_label.setStyleSheet(
            "font-size: 11px; color: #1D4ED8; font-weight: 600; line-height: 1.4; background-color: #EEF6FF; padding: 10px; border-radius: 8px; border: 1px solid #BFDBFE;"
        )
        info_layout.addWidget(help_label)

        self.training_status_label = QLabel("")
        self.training_status_label.setWordWrap(True)
        self.training_status_label.setStyleSheet("color: #334155; font-weight: 600; padding: 4px 0 8px 0;")
        info_layout.addWidget(self.training_status_label)

        self.train_btn = QPushButton("모델 학습 시작")
        self.train_btn.setFixedHeight(45)
        self.train_btn.setEnabled(False)
        self.train_btn.setStyleSheet(
            "QPushButton { background-color: #2563EB; color: white; font-weight: 700; border: none; border-radius: 10px; }"
            "QPushButton:hover { background-color: #1D4ED8; }"
        )
        self.train_btn.clicked.connect(self.on_train_clicked)
        info_layout.addWidget(self.train_btn)

        self.metrics_label = QLabel("<b>모델 성능 요약:</b><br>- 예측 정확도: N/A<br>- 평균 오차: N/A")
        self.metrics_label.setWordWrap(True)
        self.metrics_label.setStyleSheet(
            "background-color: #f8f9fa; padding: 12px; border-radius: 8px; border: 1px solid #e2e8f0;"
        )
        self.metrics_label.hide()
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

        self.perf_header_label = QLabel("상세 성능 분석 (Predicted vs Actual)")
        self.perf_header_label.setStyleSheet("font-size: 16px; font-weight: 700; color: #111827;")
        layout.addWidget(self.perf_header_label)

        self.perf_canvas = FigureCanvas(Figure(figsize=(10, 8)))
        layout.addWidget(self.perf_canvas)

        self.perf_desc_label = QLabel("* 학습이 끝나면 실제값과 예측값 비교 그래프가 여기에 표시됩니다.")
        self.perf_desc_label.setStyleSheet("color: #475569; font-weight: 600;")
        layout.addWidget(self.perf_desc_label)

        self.tabs.addTab(tab, "상세 성능 분석")
        self.render_performance_placeholder()

    def setup_inference_tab(self):
        tab = QWidget()
        self.inference_tab = tab
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(12)

        self.inference_left_frame = QWidget()
        left_frame_layout = QVBoxLayout(self.inference_left_frame)
        left_frame_layout.setContentsMargins(12, 12, 12, 12)
        left_frame_layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setStyleSheet("background: transparent; border: none;")
        left_widget = QWidget()
        left_widget.setStyleSheet("background: transparent;")
        left_panel = QVBoxLayout(left_widget)
        left_panel.setContentsMargins(0, 0, 0, 0)
        left_panel.setSpacing(12)
        scroll.setWidget(left_widget)
        left_frame_layout.addWidget(scroll)

        self.inputs = {}
        self.active_model_info = QLabel("현재 예측 모델: 아직 준비되지 않음")
        left_panel.addWidget(self.active_model_info)

        self._build_prediction_input_sections(left_panel, self.inputs)

        self.predict_btn = QPushButton("물성 예측 실행")
        self.predict_btn.setFixedHeight(46)
        self.predict_btn.setStyleSheet(
            "QPushButton { background: #0F766E; color: white; border: none; border-radius: 10px; "
            "font-weight: 700; margin-top: 4px; }"
            "QPushButton:hover { background: #0D9488; }"
        )
        self.predict_btn.clicked.connect(self.on_predict_clicked)
        left_panel.addWidget(self.predict_btn)
        left_panel.addStretch()

        self.inference_right_frame = QWidget()
        right_frame_layout = QVBoxLayout(self.inference_right_frame)
        right_frame_layout.setContentsMargins(12, 12, 12, 12)
        right_frame_layout.setSpacing(0)

        right_panel = QVBoxLayout()
        right_panel.setContentsMargins(0, 0, 0, 0)
        right_panel.setSpacing(0)
        self.inference_result_tabs = QTabWidget()
        self.inference_result_tabs.setDocumentMode(True)

        result_tab = QWidget()
        result_tab_layout = QVBoxLayout(result_tab)
        result_group = QGroupBox("예측 결과")
        result_layout = QVBoxLayout(result_group)
        self.result_display = QLabel(
            "<b>예측 준비 완료</b><br>"
            "학습된 모델이 있으면 <b>물성 예측 실행</b> 버튼으로 바로 결과를 확인할 수 있습니다."
        )
        self.result_display.setWordWrap(True)
        result_layout.addWidget(self.result_display)
        self.prediction_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        result_layout.addWidget(self.prediction_canvas)
        result_tab_layout.addWidget(result_group)
        self.inference_result_tabs.addTab(result_tab, "예측 결과")

        curve_tab = QWidget()
        curve_tab_layout = QVBoxLayout(curve_tab)
        curve_group = QGroupBox("Stress-Strain Curve")
        curve_group_layout = QVBoxLayout(curve_group)
        self.stress_strain_placeholder_label = QLabel(
            "Stress-Strain Curve 탭은 현재 자리만 준비되어 있습니다."
        )
        self.stress_strain_placeholder_label.setWordWrap(True)
        curve_group_layout.addWidget(self.stress_strain_placeholder_label)
        curve_tab_layout.addWidget(curve_group)
        self.inference_result_tabs.addTab(curve_tab, "Stress-Strain Curve")

        right_panel.addWidget(self.inference_result_tabs)
        right_frame_layout.addLayout(right_panel)

        layout.addWidget(self.inference_left_frame, 1)
        layout.addWidget(self.inference_right_frame, 1)
        self.tabs.addTab(tab, "물성 예측")

    def setup_workspace_tab(self):
        tab = QWidget()
        self.workspace_tab = tab
        tab.setStyleSheet("background: #FFFFFF;")
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        # ── 저장/불러오기 컨트롤 (이전 상단 바에서 이동) ──────────────
        ws_save_row = QHBoxLayout()
        ws_save_row.setSpacing(6)
        self.ws_name_label = QLabel("이름:")
        self.ws_name_label.setStyleSheet("font-size: 12px; color: #475569; font-weight: 600;")
        self.ws_name_input = QLineEdit()
        self.ws_name_input.setPlaceholderText("분석 저장 이름 입력 (예: 실험A)")
        self.ws_name_input.setFixedWidth(200)
        self.ws_name_input.setStyleSheet("QLineEdit { background: #FFFFFF; color: #111827; border: 1px solid #D3D7DC; border-radius: 6px; padding: 6px 10px; }")
        self.ws_save_btn = QPushButton("저장")
        self.ws_save_btn.setFixedHeight(34)
        self.ws_save_btn.setStyleSheet(
            "QPushButton { background: #E56020; color: white; border: none; border-radius: 10px; "
            "font-weight: 700; padding: 7px 16px; }"
            "QPushButton:hover { background: #F97316; }"
        )
        self.ws_save_btn.clicked.connect(self.save_workspace)
        self.ws_combo = QComboBox()
        self.ws_combo.setFixedWidth(200)
        self.ws_combo.setStyleSheet("QComboBox { background: #FFFFFF; color: #111827; border: 1px solid #D3D7DC; border-radius: 6px; padding: 5px 10px; }")
        self.ws_load_btn = QPushButton("불러오기")
        self.ws_load_btn.setFixedHeight(34)
        self.ws_load_btn.setStyleSheet(
            "QPushButton { background: #FFFFFF; color: #334155; border: 1px solid #CBD5E1; border-radius: 10px; "
            "font-weight: 700; padding: 7px 14px; }"
            "QPushButton:hover { background: #F8FAFC; border-color: #94A3B8; }"
        )
        self.ws_load_btn.clicked.connect(self.load_workspace)
        self.ws_delete_btn = QPushButton("삭제")
        self.ws_delete_btn.setFixedHeight(34)
        self.ws_delete_btn.setStyleSheet(
            "QPushButton { background: #DC2626; color: white; border: none; border-radius: 10px; "
            "font-weight: 700; padding: 7px 12px; }"
            "QPushButton:hover { background: #EF4444; }"
        )
        self.ws_delete_btn.clicked.connect(self.delete_workspace)
        ws_save_row.addWidget(self.ws_name_label)
        ws_save_row.addWidget(self.ws_name_input)
        ws_save_row.addWidget(self.ws_save_btn)
        ws_save_row.addSpacing(16)
        ws_save_row.addWidget(self.ws_combo)
        ws_save_row.addWidget(self.ws_load_btn)
        ws_save_row.addWidget(self.ws_delete_btn)
        ws_save_row.addStretch()
        layout.addLayout(ws_save_row)

        # ── 목록 헤더 ──────────────────────────────────────────────────
        header_row = QHBoxLayout()
        self.ws_list_title = QLabel("분석 저장 목록")
        self.ws_list_title.setStyleSheet("font-size: 14px; font-weight: 700; color: #111827;")
        header_row.addWidget(self.ws_list_title)
        header_row.addStretch()
        self.ws_compare_btn = QPushButton("비교 보기")
        self.ws_compare_btn.setFixedSize(98, 34)
        self.ws_compare_btn.setStyleSheet(
            "QPushButton { background: #E56020; color: white; border: none; "
            "border-radius: 17px; font-weight: 700; padding: 6px 12px; }"
            "QPushButton:hover { background: #F97316; }"
        )
        self.ws_compare_btn.clicked.connect(self._on_compare_clicked)
        header_row.addWidget(self.ws_compare_btn)
        self.ws_refresh_btn = QPushButton("목록 갱신")
        self.ws_refresh_btn.setFixedSize(98, 34)
        self.ws_refresh_btn.setStyleSheet(
            "QPushButton { background: #FFFFFF; color: #334155; border: 1px solid #CBD5E1; "
            "border-radius: 17px; font-weight: 700; padding: 6px 12px; }"
            "QPushButton:hover { background: #F8FAFC; border-color: #94A3B8; }"
        )
        self.ws_refresh_btn.clicked.connect(self.refresh_workspace_table)
        header_row.addWidget(self.ws_refresh_btn)
        layout.addLayout(header_row)

        # 상단 테이블 + 하단 상세 패널을 QSplitter로 분리
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setHandleWidth(2)

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
        self.ws_detail_widget = QWidget()
        self.ws_detail_widget.setStyleSheet("background-color: #F8FAFC; border-top: 1px solid #CBD5E1;")
        detail_layout = QHBoxLayout(self.ws_detail_widget)
        detail_layout.setContentsMargins(12, 10, 12, 10)

        # 그래프 썸네일 영역 (학습 + 상세 성능)
        thumb_col = QVBoxLayout()
        thumb_col.setSpacing(4)

        # 학습 그래프 썸네일
        self.ws_train_thumb_label = QLabel("학습 그래프")
        self.ws_train_thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ws_train_thumb_label.setStyleSheet("font-size: 11px; color: #475569; font-weight: 600;")
        thumb_col.addWidget(self.ws_train_thumb_label)
        self.ws_detail_thumb = QLabel()
        self.ws_detail_thumb.setFixedSize(200, 120)
        self.ws_detail_thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ws_detail_thumb.setStyleSheet("border: 1px solid #CBD5E1; background: white; border-radius: 8px;")
        self.ws_detail_thumb.setText("없음")
        self.ws_detail_thumb.setCursor(Qt.CursorShape.PointingHandCursor)
        self.ws_detail_thumb.mousePressEvent = self._on_thumb_clicked
        self._ws_thumb_full_path = None
        thumb_col.addWidget(self.ws_detail_thumb)

        # 상세 성능 그래프 썸네일
        self.ws_perf_thumb_label = QLabel("상세 성능")
        self.ws_perf_thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ws_perf_thumb_label.setStyleSheet("font-size: 11px; color: #475569; font-weight: 600;")
        thumb_col.addWidget(self.ws_perf_thumb_label)
        self.ws_detail_perf_thumb = QLabel()
        self.ws_detail_perf_thumb.setFixedSize(200, 120)
        self.ws_detail_perf_thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ws_detail_perf_thumb.setStyleSheet("border: 1px solid #CBD5E1; background: white; border-radius: 8px;")
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
        self.ws_detail_info.setStyleSheet("font-size: 12px; color: #334155; padding-left: 12px;")
        self.ws_detail_info.setText("저장된 분석을 클릭하면 상세 정보가 표시됩니다.")
        detail_layout.addWidget(self.ws_detail_info, 1)

        # 불러오기 버튼
        self.ws_panel_load_btn = QPushButton("분석 불러오기")
        self.ws_panel_load_btn.setFixedSize(130, 40)
        self.ws_panel_load_btn.setStyleSheet(
            "QPushButton { background: #0F766E; color: white; border: none; border-radius: 10px; "
            "font-weight: 700; padding: 8px 14px; }"
            "QPushButton:hover { background: #0D9488; }"
        )
        self.ws_panel_load_btn.clicked.connect(self._load_selected_ws)
        detail_layout.addWidget(self.ws_panel_load_btn, 0, Qt.AlignmentFlag.AlignBottom)

        splitter.addWidget(self.ws_detail_widget)
        splitter.setSizes([400, 180])
        layout.addWidget(splitter)

        self.ws_hint_label = QLabel("※ 행 클릭 → 상세 보기  |  더블클릭 → 바로 불러오기  |  Ctrl+클릭으로 여러 행 선택 후 [비교 보기] 클릭")
        self.ws_hint_label.setStyleSheet("color: #475569; font-size: 11px; font-weight: 600; margin-top: 4px;")
        layout.addWidget(self.ws_hint_label)

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
        self.data_engine.configure_quality_rules(
            missing_strategy=missing_map.get(self.missing_combo.currentIndex(), "mean"),
            outlier_strategy=outlier_map.get(self.outlier_combo.currentIndex(), "clip"),
            invalid_type_strategy=invalid_type_map.get(self.invalid_type_combo.currentIndex(), "coerce"),
            iqr_factor=self.iqr_spin.value(),
            input_feature_mode="combined",
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
        colors = self._theme()
        self.canvas.axes.clear()
        self.canvas.axes.axis("off")
        self.canvas.axes.set_facecolor(colors["panel_bg"])
        self.canvas.fig.patch.set_facecolor(colors["panel_bg"])
        self.canvas.axes.text(0.5, 0.58, "2번 탭에서 모델 학습 결과가 여기에 표시됩니다.", ha="center", va="center", fontsize=14, color=colors["text_sec"], transform=self.canvas.axes.transAxes)
        self.canvas.axes.text(0.5, 0.42, "먼저 1번 탭에서 전처리를 실행해 주세요.", ha="center", va="center", fontsize=11, color=colors["text_label"], transform=self.canvas.axes.transAxes)
        self.canvas.draw()

    def render_performance_placeholder(self):
        colors = self._theme()
        self.perf_canvas.figure.clear()
        ax = self.perf_canvas.figure.add_subplot(111)
        ax.axis("off")
        ax.set_facecolor(colors["panel_bg"])
        self.perf_canvas.figure.patch.set_facecolor(colors["panel_bg"])
        ax.text(0.5, 0.56, "모델 학습이 끝나면 상세 성능 분석 그래프가 여기에 표시됩니다.", ha="center", va="center", fontsize=14, color=colors["text_sec"], transform=ax.transAxes)
        ax.text(0.5, 0.40, "실제값과 예측값이 얼마나 비슷한지 특성별로 확인할 수 있습니다.", ha="center", va="center", fontsize=11, color=colors["text_label"], transform=ax.transAxes)
        self.perf_canvas.figure.tight_layout()
        self.perf_canvas.draw()

    def populate_feature_selection_table(self, reset_selection=False):
        available_columns = self.data_engine.get_available_training_columns(include_engineered=True)
        previous_selection = [] if reset_selection else self.data_engine.get_selected_training_columns(default_to_all=False)
        if reset_selection or self.data_engine.selected_training_columns is None:
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
        selected_columns = self.data_engine.get_selected_training_columns(default_to_all=False)

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
            self._sb_status.setStyleSheet("color: #E56020; font-size: 11px; font-weight: 700; padding: 0 10px;")
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
            self.training_status_label.setText("상태: 합금 지표 생성 완료. 이제 선택한 학습 컬럼으로 모델을 학습할 수 있습니다.")
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
        intro.setStyleSheet("font-size: 13px; color: #334155; font-weight: 600; padding-bottom: 4px;")
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

        selected_columns = self.data_engine.get_selected_training_columns(default_to_all=False)
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
        self._sb_status.setStyleSheet("color: #58C472; font-size: 11px; font-weight: 700; padding: 0 10px;")

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

    def _render_prediction_chart(self, canvas, mean, std):
        canvas.fig.clear()
        ax1 = canvas.fig.add_subplot(111)
        labels = ["Yield", "UTS", "Elong.", "Area Red."]
        x = np.arange(len(labels))

        stress_vals = [mean[0], mean[1], 0, 0]
        stress_errs = [1.96 * std[0], 1.96 * std[1], 0, 0]
        ax1.bar(x[:2], stress_vals[:2], yerr=stress_errs[:2], capsize=10, color=["#3498db", "#e74c3c"])
        ax1.set_ylabel("Stress (MPa)", color="#3498db")
        ax1.tick_params(axis="y", colors="#3498db")

        ax2 = ax1.twinx()
        duct_vals = [0, 0, mean[2], mean[3]]
        duct_errs = [0, 0, 1.96 * std[2], 1.96 * std[3]]
        ax2.bar(x[2:], duct_vals[2:], yerr=duct_errs[2:], capsize=10, color=["#2ecc71", "#f39c12"])
        ax2.set_ylabel("Percentage (%)", color="#2ecc71")
        ax2.tick_params(axis="y", colors="#2ecc71")

        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.set_title("예측 물성 결과")
        canvas.fig.tight_layout()
        canvas.draw()

    def _run_prediction(self, model_engine, data_engine, inputs, result_label, canvas, result_tabs=None):
        if not model_engine or not data_engine:
            result_label.setText(
                "<b>먼저 모델을 준비해 주세요.</b><br>"
                "학습을 완료하거나 저장된 모델을 불러온 뒤 예측을 실행할 수 있습니다."
            )
            return None

        input_dict = {key: widget.text() for key, widget in inputs.items()}
        scaled_input = data_engine.get_inference_data(input_dict)
        mean_scaled, std_scaled = model_engine.predict(scaled_input.astype(np.float32))

        mean = data_engine.scaler_y.inverse_transform(mean_scaled)[0]
        std = std_scaled[0] * data_engine.scaler_y.scale_

        result_text = (
            f"<b>강도 예측 결과</b><br>"
            f"0.2% 항복강도: <b>{mean[0]:.1f} ± {std[0]:.1f} MPa</b><br>"
            f"인장강도(UTS): <b>{mean[1]:.1f} ± {std[1]:.1f} MPa</b><br><br>"
            f"<b>연성 예측 결과</b><br>"
            f"연신율: <b>{mean[2]:.1f} ± {std[2]:.1f} %</b><br>"
            f"단면감소율: <b>{mean[3]:.1f} ± {std[3]:.1f} %</b>"
        )
        result_label.setText(result_text)
        self._render_prediction_chart(canvas, mean, std)

        if result_tabs is not None:
            result_tabs.setCurrentIndex(0)

        return {
            "yield_stress": round(float(mean[0]), 2),
            "uts": round(float(mean[1]), 2),
            "elongation": round(float(mean[2]), 2),
            "area_reduction": round(float(mean[3]), 2),
        }

    def on_pretrained_predict_clicked(self):
        try:
            self._run_prediction(
                self.pretrained_model_engine,
                self.pretrained_data_engine,
                self.pretrained_inputs,
                self.pretrained_result_display,
                self.pretrained_prediction_canvas,
                self.pretrained_result_tabs,
            )
        except Exception as exc:
            self.pretrained_result_display.setText(f"<b>사전학습 모델 예측 중 오류가 발생했습니다.</b><br>{exc}")

    def on_predict_clicked(self):
        try:
            results = self._run_prediction(
                self.model_engine,
                self.data_engine,
                self.inputs,
                self.result_display,
                self.prediction_canvas,
                self.inference_result_tabs,
            )
            if results is None:
                return

            auto_folder = os.path.join("workspaces", "auto_save")
            if os.path.exists(auto_folder):
                self.prediction_canvas.fig.savefig(
                    os.path.join(auto_folder, "prediction.png"), dpi=200, bbox_inches="tight"
                )

            self.append_log({
                "type": "예측",
                "model": self.model_type,
                "inputs": {k: v.text() for k, v in self.inputs.items()},
                "results": results,
            })
        except Exception as exc:
            self.result_display.setText(f"<b>예측 중 오류가 발생했습니다.</b><br>{exc}")

    def prepare_pretrained_model(self):
        try:
            model_info = self._load_or_train_pretrained_model()
            self.pretrained_model_engine = model_info["model_engine"]
            self.pretrained_data_engine = model_info["data_engine"]
            self.pretrained_model_type = model_info["model_type"]
            self.pretrained_metrics = model_info["metrics"]

            name_map = {"RF": "Random Forest", "GBM": "Gradient Boosting", "MLP": "Neural Network", "TFP": "TFP"}
            model_name = name_map.get(self.pretrained_model_type, self.pretrained_model_type)
            self.pretrained_active_model_info.setText(
                f"사용 중인 모델: {model_name} | 평균 R2 {self.pretrained_metrics['r2_avg']:.3f} | 평균 MAE {self.pretrained_metrics['mae_avg']:.3f}"
            )
            self.pretrained_active_model_info.hide()
            self.pretrained_predict_btn.setEnabled(True)
        except Exception as exc:
            self.pretrained_predict_btn.setEnabled(False)
            self.pretrained_result_display.setText(
                "<b>사전학습 모델을 준비하지 못했습니다.</b><br>"
                f"{exc}"
            )
            self.pretrained_active_model_info.setText("사용 중인 모델: 준비되지 않음")
            self.pretrained_active_model_info.hide()

    def _load_or_train_pretrained_model(self):
        models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
        os.makedirs(models_dir, exist_ok=True)

        model_path = os.path.join(models_dir, "pretrained_material_model.pkl")
        data_engine_path = os.path.join(models_dir, "pretrained_data_engine.pkl")
        meta_path = os.path.join(models_dir, "pretrained_material_model_meta.json")
        dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "STMECH_AUS_SS.xls"))

        if os.path.exists(model_path) and os.path.exists(data_engine_path) and os.path.exists(meta_path):
            pretrained_engine = joblib.load(data_engine_path)
            selected_columns = pretrained_engine.get_selected_training_columns()
            if "Fe" in selected_columns:
                pretrained_model = ModelEngine(model_type="RF", output_dim=4)
                pretrained_model.load(model_path)
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                return {
                    "model_engine": pretrained_model,
                    "data_engine": pretrained_engine,
                    "model_type": meta.get("model_type", pretrained_model.model_type),
                    "metrics": meta,
                }

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"사전학습 데이터 파일을 찾을 수 없습니다: {dataset_path}")

        from sklearn.metrics import mean_absolute_error, r2_score

        best_result = None
        for model_type in ["RF", "GBM", "MLP"]:
            data_engine = DataEngine(dataset_path)
            data_engine.load_data(include_engineered=True)
            data_engine.set_selected_training_columns(
                data_engine.get_available_training_columns(include_engineered=True)
            )

            X_train, X_test, y_train, _, _, y_raw_test = data_engine.preprocess_data()
            if len(X_train) == 0:
                continue

            model_engine = ModelEngine(model_type=model_type, output_dim=y_train.shape[1], max_iter=2000)
            model_engine.train(X_train, y_train)
            mean_scaled, _ = model_engine.predict(X_test)
            y_pred = data_engine.inverse_transform_y(mean_scaled)

            r2 = r2_score(y_raw_test, y_pred, multioutput="raw_values")
            mae = mean_absolute_error(y_raw_test, y_pred, multioutput="raw_values")
            metrics = {
                "model_type": model_type,
                "data_file": os.path.basename(dataset_path),
                "r2_avg": round(float(np.mean(r2)), 4),
                "mae_avg": round(float(np.mean(mae)), 4),
                "r2_per_target": [round(float(v), 4) for v in r2],
                "mae_per_target": [round(float(v), 4) for v in mae],
            }

            if best_result is None or metrics["r2_avg"] > best_result["metrics"]["r2_avg"]:
                best_result = {
                    "model_engine": model_engine,
                    "data_engine": data_engine,
                    "model_type": model_type,
                    "metrics": metrics,
                }

        if best_result is None:
            raise ValueError("사전학습 모델을 만들 수 있는 학습 데이터가 없습니다.")

        best_result["model_engine"].save(model_path)
        joblib.dump(best_result["data_engine"], data_engine_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(best_result["metrics"], f, ensure_ascii=False, indent=2)
        return best_result

    def update_active_model_display(self):
        name_map = {"RF": "Random Forest", "GBM": "Gradient Boosting", "MLP": "Neural Network", "TFP": "TFP"}
        self.active_model_info.setText(f"현재 예측 모델: {name_map.get(self.model_type, self.model_type)}")
        if self.pretrained_model_type and hasattr(self, "pretrained_active_model_info"):
            metrics = self.pretrained_metrics or {}
            model_name = name_map.get(self.pretrained_model_type, self.pretrained_model_type)
            if metrics:
                self.pretrained_active_model_info.setText(
                    f"사용 중인 모델: {model_name} | 평균 R2 {metrics.get('r2_avg', 0):.3f} | 평균 MAE {metrics.get('mae_avg', 0):.3f}"
                )
            else:
                self.pretrained_active_model_info.setText(f"사용 중인 모델: {model_name}")
        self._apply_theme_colors()

    # ================================================================
    # [AUTO SAVE] ??? ??? ??workspaces/auto_save/ ???????? ????(1??? ???, ?????)
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
                "training_input_combo": 0,
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
                "training_input_combo": 0,
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
            self.missing_combo.setCurrentIndex(pre.get("missing_combo", 0))
            self.outlier_combo.setCurrentIndex(pre.get("outlier_combo", 0))
            self.invalid_type_combo.setCurrentIndex(pre.get("invalid_type_combo", 0))
            self.iqr_spin.setValue(pre.get("iqr_spin", 1.5))
            if hasattr(self, "training_input_combo"):
                self.training_input_combo.setCurrentIndex(0)
            self.preprocessing_ready = pre.get("preprocessing_ready", False)
            self.train_btn.setEnabled(self.preprocessing_ready)
            self.go_to_training_btn.setEnabled(self.preprocessing_ready)
            self.missing_combo.blockSignals(False)
            self.outlier_combo.blockSignals(False)
            self.invalid_type_combo.blockSignals(False)
            self.iqr_spin.blockSignals(False)

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
        <h3>학습 컬럼 선택</h3>
        <p>학습에 사용할 변수는 <b>학습 컬럼 선택</b> 탭에서 고릅니다. 체크한 컬럼만 모델 학습과 예측에 사용됩니다.</p>
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

