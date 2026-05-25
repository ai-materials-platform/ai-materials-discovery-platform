import os

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.gui.widgets import MplCanvas


class SettingsPanelMixin:
    def _create_left_panel(self):
        self._tree_section_title_lbls = []

        panel = QWidget()
        panel.setMinimumWidth(160)
        panel.setMaximumWidth(240)
        panel.setStyleSheet("background: #FFFFFF;")
        self._panel_widgets.append(panel)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QLabel("분석 기록 탐색기")
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

    def _create_settings_panel(self):
        panel = QWidget()
        panel.setMinimumWidth(280)
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
        self._setup_fe_auto_update(self.inputs)

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
        curve_group_layout.addWidget(
            self._create_curve_info_panel(
                "stress_strain_placeholder_label",
                "stress_strain_legend_card",
            )
        )
        self.stress_strain_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        curve_group_layout.addWidget(self.stress_strain_canvas)
        curve_tab_layout.addWidget(curve_group)
        self.inference_result_tabs.addTab(curve_tab, "Stress-Strain Curve")

        simulation_tab = self._create_simulation_tab("user")
        self.inference_result_tabs.addTab(simulation_tab, "Simulation")

        right_panel.addWidget(self.inference_result_tabs)
        right_frame_layout.addLayout(right_panel)
        self.render_prediction_placeholder(
            self.prediction_canvas,
            "물성 예측 결과",
            title_fontsize=11.4,
            body_fontsize=9.0,
        )
        self.render_stress_strain_placeholder(
            self.stress_strain_canvas,
            self.stress_strain_placeholder_label,
        )

        self.inference_left_frame.setMinimumWidth(300)
        self.inference_left_frame.setMaximumWidth(420)
        layout.addWidget(self.inference_left_frame, 0)
        layout.addWidget(self.inference_right_frame, 1)
        self.tabs.addTab(tab, "물성 예측")

    def setup_workspace_tab(self):
        tab = QWidget()
        self.workspace_tab = tab
        tab.setStyleSheet("background: #FFFFFF;")
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        ws_save_row = QHBoxLayout()
        ws_save_row.setSpacing(6)
        self.ws_name_label = QLabel("이름:")
        self.ws_name_label.setStyleSheet("font-size: 12px; color: #475569; font-weight: 600;")
        self.ws_name_input = QLineEdit()
        self.ws_name_input.setPlaceholderText("분석 기록 이름 입력 (예: 실험A)")
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

        header_row = QHBoxLayout()
        self.ws_list_title = QLabel("분析 기록 목록")
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

        self.ws_table = QTableWidget()
        self.ws_table.setColumnCount(7)
        self.ws_table.setHorizontalHeaderLabels(["이름", "모델", "저장 날짜", "초기값", "회복_구간", "복원불가_구간", "끊기는_구간"])
        self.ws_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.ws_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.ws_table.setSelectionMode(QTableWidget.SelectionMode.MultiSelection)
        self.ws_table.setAlternatingRowColors(True)
        self.ws_table.verticalHeader().setVisible(False)
        self.ws_table.horizontalHeader().setStretchLastSection(True)
        self.ws_table.setColumnWidth(0, 160)
        self.ws_table.setColumnWidth(1, 130)
        self.ws_table.setColumnWidth(2, 160)
        self.ws_table.setColumnWidth(3, 120)
        self.ws_table.setColumnWidth(4, 140)
        self.ws_table.setColumnWidth(5, 150)
        self.ws_table.setColumnWidth(6, 140)
        self.ws_table.cellClicked.connect(self._on_ws_table_clicked)
        self.ws_table.cellDoubleClicked.connect(self._on_ws_table_double_clicked)
        layout.addWidget(self.ws_table)

        self.ws_hint_label = QLabel("※ 더블클릭 → 바로 불러오기  |  Ctrl+클릭으로 여러 행 선택 후 [비교 보기] 클릭")
        self.ws_hint_label.setStyleSheet("color: #475569; font-size: 11px; font-weight: 600; margin-top: 4px;")
        layout.addWidget(self.ws_hint_label)

        self._workspace_widget = tab
        self.refresh_workspace_table()
