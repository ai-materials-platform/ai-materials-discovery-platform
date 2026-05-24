import matplotlib.pyplot as plt
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from src.gui.constants import APP_FONT_SIZE, GLOBAL_QSS
from src.gui.widgets import MplCanvas, PredictionGuideOverlay
from src.gui.mixins.process_condition_mixin import ProcessConditionPanel


class UISetupMixin:
    def init_ui(self):
        self.setStyleSheet(GLOBAL_QSS)
        self._apply_ui_font()

        mb = self.menuBar()
        mb.setStyleSheet(
            "QMenuBar { background: #252525; color: #D7DCE3; font-size: 12px; padding: 3px 6px; border-bottom: 1px solid #363636; }"
            "QMenuBar::item { background: transparent; padding: 5px 10px; }"
            "QMenuBar::item:selected { background: #3A4048; color: #FFFFFF; }"
        )
        file_menu = mb.addMenu("파일")
        file_menu.addAction("분석 기록 저장", self._save_workspace_from_menu)
        file_menu.addAction("분석 기록 불러오기", self._open_workspace_dialog)
        for name in ["편집", "보기", "데이터", "분석", "도구", "도움말"]:
            mb.addMenu(name)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        root_layout.addWidget(self._create_toolbar())

        self.main_mode_stack = QStackedWidget()
        root_layout.addWidget(self.main_mode_stack, 1)

        self.material_prediction_page = self._create_material_prediction_page()
        self.main_mode_stack.addWidget(self.material_prediction_page)

        self.user_page = QWidget()
        user_layout = QVBoxLayout(self.user_page)
        user_layout.setContentsMargins(0, 0, 0, 0)
        user_layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(2)

        splitter.addWidget(self._create_left_panel())
        splitter.addWidget(self._create_settings_panel())

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        splitter.addWidget(self.tabs)
        splitter.setSizes([190, 300, 860])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 0)
        splitter.setStretchFactor(2, 1)

        user_layout.addWidget(splitter, 1)
        self.main_mode_stack.addWidget(self.user_page)
        self._switch_main_mode(0)

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
        self.setup_process_condition_tab()
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
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._mp_mode_tabs = QTabWidget()
        self._mp_mode_tabs.setDocumentMode(True)

        # ── 탭 1: 사전학습 예측 ──
        pretrained_tab = QWidget()
        pretrained_layout = QVBoxLayout(pretrained_tab)
        pretrained_layout.setContentsMargins(12, 12, 12, 12)
        pretrained_layout.setSpacing(12)

        top_row = QHBoxLayout()
        top_row.addStretch()
        guide_btn = QPushButton("사용 가이드")
        guide_btn.setFixedHeight(30)
        guide_btn.setStyleSheet(
            "QPushButton { background: #EEF6FF; color: #1D4ED8; border: 1px solid #BFDBFE; "
            "border-radius: 14px; font-size: 11px; font-weight: 700; padding: 0 14px; }"
            "QPushButton:hover { background: #DBEAFE; }"
        )
        guide_btn.clicked.connect(lambda: self._show_prediction_guide(page))
        top_row.addWidget(guide_btn)
        pretrained_layout.addLayout(top_row)

        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        content_splitter.setChildrenCollapsible(False)
        content_splitter.setHandleWidth(4)

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
        self._setup_fe_auto_update(self.pretrained_inputs)
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
        input_card.setMinimumWidth(280)
        content_splitter.addWidget(input_card)

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
        curve_layout.addWidget(
            self._create_curve_info_panel(
                "pretrained_curve_placeholder",
                "pretrained_curve_legend_card",
            )
        )
        self.pretrained_curve_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        curve_layout.addWidget(self.pretrained_curve_canvas)
        self.pretrained_result_tabs.addTab(curve_tab, "Stress-Strain Curve")

        simulation_tab = self._create_simulation_tab("pretrained")
        self.pretrained_result_tabs.addTab(simulation_tab, "Simulation")

        result_layout.addWidget(self.pretrained_result_tabs)
        self.render_prediction_placeholder(
            self.pretrained_prediction_canvas,
            "사전학습 예측 결과",
            title_fontsize=12.5,
            body_fontsize=9.8,
        )
        self.render_stress_strain_placeholder(
            self.pretrained_curve_canvas,
            self.pretrained_curve_placeholder,
        )

        self.pretrained_status_label = QLabel("")
        self.pretrained_status_label.setWordWrap(True)
        self.pretrained_status_label.hide()

        self.pretrained_model_summary_label = QLabel("")
        self.pretrained_model_summary_label.setWordWrap(True)
        self.pretrained_model_summary_label.hide()
        content_splitter.addWidget(result_card)
        content_splitter.setStretchFactor(0, 0)
        content_splitter.setStretchFactor(1, 1)
        content_splitter.setSizes([340, 900])

        pretrained_layout.addWidget(content_splitter, 1)
        self._mp_mode_tabs.addTab(pretrained_tab, "사전학습 예측")

        # ── 탭 2: 공정 조건 예측기 ──
        self._mp_mode_tabs.addTab(ProcessConditionPanel(), "공정 조건 예측기")

        outer.addWidget(self._mp_mode_tabs, 1)
        return page

    def _show_prediction_guide(self, _page):
        steps = [
            {
                "widget": self.pretrained_inputs.get(
                    list(self.pretrained_inputs.keys())[0]
                ).parent().parent().parent().parent()
                if self.pretrained_inputs else None,
                "text": "① 합금 조성 입력\n\n각 원소의 wt% 값을 입력하세요.\nFe는 보통 96% 이상, 나머지는 소량.\n\n기본값은 오스테나이트계 스테인리스강 기준입니다.",
            },
            {
                "widget": self.pretrained_predict_btn,
                "text": "② 예측 실행\n\n버튼 클릭 시 RF·GBM·MLP·TFP\n4개 AI 모델이 물성을 예측합니다.\n\n평균값과 불확실도를 함께 제공합니다.",
            },
            {
                "widget": self.pretrained_result_tabs,
                "text": "③ 예측 결과 탭\n\nYield Stress / UTS / 연신율 /\n단면감소율 예측값을 확인합니다.\n\n오차 범위가 좁을수록 신뢰도가 높습니다.",
                "on_show": lambda: self.pretrained_result_tabs.setCurrentIndex(0),
            },
            {
                "widget": self.pretrained_result_tabs,
                "text": "④ Stress-Strain Curve 탭\n\n예측 물성 기반 응력-변형률 곡선.\n• 초록: 탄성 구간 (복원 가능)\n• 주황: 소성 구간 (영구 변형)\n• 빨강: 네킹→파단 구간",
                "on_show": lambda: self.pretrained_result_tabs.setCurrentIndex(1),
            },
            {
                "widget": self.pretrained_result_tabs,
                "text": "⑤ Simulation 탭\n\n핸들 드래그로 인장 하중을 조절하며\n탄성/소성 변형을 실시간 확인.\n\nStress-Strain 그래프에 현재 위치가 표시됩니다.",
                "on_show": lambda: self.pretrained_result_tabs.setCurrentIndex(2),
            },
        ]
        if not self.pretrained_inputs:
            steps[0]["widget"] = None
        else:
            try:
                first_key = list(self.pretrained_inputs.keys())[0]
                le = self.pretrained_inputs[first_key]
                steps[0]["widget"] = le.parent().parent().parent().parent()
            except Exception:
                steps[0]["widget"] = None

        cw = self.centralWidget()
        toolbar_h = self._toolbar_widget.height()
        overlay = PredictionGuideOverlay(steps, parent=cw, toolbar_h=toolbar_h)
        overlay.raise_()
        overlay.show()

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
            label.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
            line_edit = QLineEdit()
            line_edit.setText(value)
            line_edit.deselect()
            line_edit.setCursorPosition(0)
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
        label_color = "#555555" if not self._dark_mode else c['text_sec']
        for label in self._prediction_input_labels:
            label.setStyleSheet(
                f"color: {label_color}; font-size: 12px; font-weight: 600; "
                "padding-right: 4px; background: transparent;"
            )
        for field in self._prediction_input_fields:
            field.setStyleSheet(line_edit_style)
        if hasattr(self, "_fe_readonly_fields"):
            for fe_field in self._fe_readonly_fields:
                self._apply_fe_readonly_style(fe_field)

    def _apply_fe_readonly_style(self, field):
        if self._dark_mode:
            bg, color, border = "#1E2229", "#64748B", "#3A4048"
        else:
            bg, color, border = "#F1F5F9", "#64748B", "#CBD5E1"
        field.setStyleSheet(
            f"QLineEdit {{ background: {bg}; color: {color}; border: 1px solid {border}; "
            "border-radius: 8px; padding: 7px 10px; }"
            f"QLineEdit:focus {{ border-color: {border}; }}"
        )

    _COMPOSITION_KEYS = ["Fe", "C", "Si", "Mn", "P", "S", "Ni", "Cr", "Mo", "Cu", "V", "N", "Nb", "Ti", "B", "Al"]

    def _setup_fe_auto_update(self, input_store):
        fe_field = input_store.get("Fe")
        if not fe_field:
            return
        fe_field.setReadOnly(True)
        if not hasattr(self, "_fe_readonly_fields"):
            self._fe_readonly_fields = []
        self._fe_readonly_fields.append(fe_field)
        self._apply_fe_readonly_style(fe_field)

        other_keys = [k for k in self._COMPOSITION_KEYS if k != "Fe" and k in input_store]

        def _update_fe():
            total = 0.0
            for key in other_keys:
                try:
                    total += float(input_store[key].text())
                except ValueError:
                    pass
            fe_val = 100.0 - total
            fe_str = f"{fe_val:.4f}".rstrip("0")
            if fe_str.endswith("."):
                fe_str += "0"
            fe_field.blockSignals(True)
            fe_field.setText(fe_str)
            fe_field.blockSignals(False)

        for key in other_keys:
            input_store[key].textChanged.connect(_update_fe)
        _update_fe()

    def resizeEvent(self, event):
        from PyQt6.QtWidgets import QMainWindow
        QMainWindow.resizeEvent(self, event)

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
