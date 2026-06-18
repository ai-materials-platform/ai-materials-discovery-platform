import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

_SLIDER_STEPS = 1000

_COLUMN_RANGES = {
    "C":  (0.01, 0.30),
    "Si": (0.10, 3.00),
    "Mn": (0.50, 5.00),
    "P":  (0.001, 0.05),
    "S":  (0.001, 0.03),
    "Ni": (0.10, 20.0),
    "Cr": (14.0, 30.0),
    "Mo": (0.00, 6.00),
    "Cu": (0.00, 4.00),
    "V":  (0.00, 1.00),
    "N":  (0.001, 0.40),
    "Nb": (0.00, 0.50),
    "Ti": (0.00, 0.50),
    "B":  (0.0001, 0.005),
    "Al": (0.001, 0.10),
    "Solution_treatment_temperature": (900.0, 1200.0),
    "Solution_treatment_time(s)":     (600.0, 14400.0),
    "Grains mm-2":                    (100.0, 2000.0),
    "Temperature (K)":                (200.0, 1300.0),
}

_COLUMN_LABELS = {
    "Solution_treatment_temperature": "용체화 처리 온도 (°C)",
    "Solution_treatment_time(s)":     "용체화 처리 시간 (s)",
    "Grains mm-2":                    "결정립 수 (mm⁻²)",
    "Temperature (K)":                "시험 온도 (K)",
}

_COMP_COLS = ["C", "Si", "Mn", "P", "S", "Ni", "Cr", "Mo", "Cu", "V", "N", "Nb", "Ti", "B", "Al"]
_PROC_COLS = [
    "Solution_treatment_temperature",
    "Solution_treatment_time(s)",
    "Grains mm-2",
    "Temperature (K)",
]


class StrainExploreDialog(QDialog):
    def __init__(self, model_engine, data_engine, base_input_dict, build_fn, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Stress-Strain 상세 탐색기")
        self.resize(1100, 660)
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowCloseButtonHint
        )

        self._model_engine = model_engine
        self._data_engine = data_engine
        self._base_input = dict(base_input_dict)
        self._build_fn = build_fn
        self._base_mean = self._compute_base_mean()

        self._update_timer = QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(80)
        self._update_timer.timeout.connect(self._update_curve)

        self._avail_cols = [c for c in (_COMP_COLS + _PROC_COLS) if c in self._base_input]

        self._setup_ui()
        self._on_column_changed()

    # ── UI 구성 ──────────────────────────────────────────────────────────────

    def _setup_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(4)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter)

        splitter.addWidget(self._build_ctrl_panel())
        splitter.addWidget(self._build_chart_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([290, 810])

    def _build_ctrl_panel(self):
        ctrl = QWidget()
        ctrl.setMinimumWidth(260)
        ctrl.setMaximumWidth(320)
        ctrl.setStyleSheet("background: #F8FAFC; border-right: 1px solid #E2E8F0;")
        layout = QVBoxLayout(ctrl)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)

        title = QLabel("조성 / 공정 탐색기")
        title.setStyleSheet("font-size: 14px; font-weight: 700; color: #111827;")
        layout.addWidget(title)

        desc = QLabel("컬럼을 선택하고 슬라이더를 드래그하면\nStress-Strain Curve가 실시간 업데이트됩니다.")
        desc.setWordWrap(True)
        desc.setStyleSheet("font-size: 11px; color: #64748B;")
        layout.addWidget(desc)

        # 컬럼 선택
        col_box = QGroupBox("탐색 컬럼")
        col_layout = QVBoxLayout(col_box)
        col_layout.setContentsMargins(10, 10, 10, 10)

        comp_label = QLabel("▸ 합금 조성 (wt%)")
        comp_label.setStyleSheet("font-size: 10px; color: #94A3B8; font-weight: 600;")
        col_layout.addWidget(comp_label)

        self._col_combo = QComboBox()
        for c in self._avail_cols:
            if c in _COMP_COLS:
                self._col_combo.addItem(c, c)
        self._col_combo.insertSeparator(self._col_combo.count())
        sep_label_idx = self._col_combo.count()
        proc_items = [c for c in self._avail_cols if c in _PROC_COLS]
        if proc_items:
            for c in proc_items:
                self._col_combo.addItem(_COLUMN_LABELS.get(c, c), c)
        self._col_combo.currentIndexChanged.connect(self._on_column_changed)
        col_layout.addWidget(self._col_combo)
        layout.addWidget(col_box)

        # 범위 설정
        range_box = QGroupBox("값 범위")
        range_layout = QHBoxLayout(range_box)
        range_layout.setSpacing(6)

        self._min_spin = QDoubleSpinBox()
        self._max_spin = QDoubleSpinBox()
        for spin in (self._min_spin, self._max_spin):
            spin.setDecimals(4)
            spin.setRange(0.0, 99999)
            spin.setFixedHeight(30)

        self._min_spin.valueChanged.connect(self._on_min_changed)
        self._max_spin.valueChanged.connect(self._on_max_changed)

        range_layout.addWidget(QLabel("최소"))
        range_layout.addWidget(self._min_spin, 1)
        range_layout.addWidget(QLabel("최대"))
        range_layout.addWidget(self._max_spin, 1)
        layout.addWidget(range_box)

        # 슬라이더
        slider_box = QGroupBox("현재 값")
        slider_layout = QVBoxLayout(slider_box)
        slider_layout.setContentsMargins(10, 10, 10, 10)
        slider_layout.setSpacing(6)

        self._val_label = QLabel("—")
        self._val_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._val_label.setStyleSheet(
            "font-size: 22px; font-weight: 700; color: #1E293B; "
            "padding: 4px 0;"
        )
        slider_layout.addWidget(self._val_label)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, _SLIDER_STEPS)
        self._slider.setValue(_SLIDER_STEPS // 2)
        self._slider.setStyleSheet(
            "QSlider::groove:horizontal { height: 6px; background: #E2E8F0; border-radius: 3px; }"
            "QSlider::handle:horizontal { width: 18px; height: 18px; margin: -6px 0; "
            "background: #1E293B; border-radius: 9px; }"
            "QSlider::sub-page:horizontal { background: #1E293B; border-radius: 3px; }"
        )
        self._slider.valueChanged.connect(self._on_slider_changed)
        slider_layout.addWidget(self._slider)

        range_row = QHBoxLayout()
        self._slider_min_lbl = QLabel("—")
        self._slider_max_lbl = QLabel("—")
        for lbl in (self._slider_min_lbl, self._slider_max_lbl):
            lbl.setStyleSheet("font-size: 10px; color: #94A3B8;")
        range_row.addWidget(self._slider_min_lbl)
        range_row.addStretch()
        range_row.addWidget(self._slider_max_lbl)
        slider_layout.addLayout(range_row)
        layout.addWidget(slider_box)

        # 예측 결과 표시
        self._result_label = QLabel("예측 결과가 여기에 표시됩니다.")
        self._result_label.setWordWrap(True)
        self._result_label.setTextFormat(Qt.TextFormat.RichText)
        self._result_label.setStyleSheet(
            "background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; "
            "padding: 12px; font-size: 12px; color: #334155; line-height: 1.6;"
        )
        layout.addWidget(self._result_label)

        layout.addStretch()

        reset_btn = QPushButton("기준값으로 초기화")
        reset_btn.setFixedHeight(34)
        reset_btn.setStyleSheet(
            "QPushButton { background: #F1F5F9; color: #475569; border: 1px solid #CBD5E1; "
            "border-radius: 8px; font-size: 11px; font-weight: 600; }"
            "QPushButton:hover { background: #E2E8F0; }"
        )
        reset_btn.clicked.connect(self._reset_to_base)
        layout.addWidget(reset_btn)

        return ctrl

    def _build_chart_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        self._canvas_fig = Figure(figsize=(7, 5), dpi=100)
        self._canvas_fig.patch.set_facecolor("#FFFFFF")
        self._canvas = FigureCanvas(self._canvas_fig)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._canvas)
        return panel

    # ── 이벤트 핸들러 ────────────────────────────────────────────────────────

    def _on_column_changed(self):
        col = self._col_combo.currentData()
        if not col:
            return
        try:
            cur = float(self._base_input.get(col, 0))
        except (ValueError, TypeError):
            cur = 0.0

        rng = _COLUMN_RANGES.get(col)
        if rng:
            lo, hi = rng
        else:
            lo = max(0.0, cur * 0.5)
            hi = cur * 1.5 if cur > 0 else lo + 1.0
            if hi - lo < 0.01:
                hi = lo + 1.0

        self._min_spin.blockSignals(True)
        self._max_spin.blockSignals(True)
        self._min_spin.setValue(lo)
        self._max_spin.setValue(hi)
        self._min_spin.blockSignals(False)
        self._max_spin.blockSignals(False)

        frac = (cur - lo) / (hi - lo) if hi > lo else 0.5
        frac = max(0.0, min(1.0, frac))
        self._slider.blockSignals(True)
        self._slider.setValue(int(frac * _SLIDER_STEPS))
        self._slider.blockSignals(False)

        self._slider_min_lbl.setText(f"{lo:.4g}")
        self._slider_max_lbl.setText(f"{hi:.4g}")
        self._val_label.setText(f"{cur:.4g}")
        self._update_timer.start()

    def _on_min_changed(self):
        lo = self._min_spin.value()
        hi = self._max_spin.value()
        if lo >= hi:
            self._max_spin.blockSignals(True)
            self._max_spin.setValue(lo + max(lo * 0.01, 0.0001))
            self._max_spin.blockSignals(False)
        self._slider_min_lbl.setText(f"{self._min_spin.value():.4g}")
        self._slider_max_lbl.setText(f"{self._max_spin.value():.4g}")
        self._on_slider_changed()

    def _on_max_changed(self):
        lo = self._min_spin.value()
        hi = self._max_spin.value()
        if hi <= lo:
            self._min_spin.blockSignals(True)
            self._min_spin.setValue(max(0.0, hi - max(hi * 0.01, 0.0001)))
            self._min_spin.blockSignals(False)
        self._slider_min_lbl.setText(f"{self._min_spin.value():.4g}")
        self._slider_max_lbl.setText(f"{self._max_spin.value():.4g}")
        self._on_slider_changed()

    def _on_slider_changed(self):
        lo = self._min_spin.value()
        hi = self._max_spin.value()
        if hi <= lo:
            return
        val = lo + (self._slider.value() / _SLIDER_STEPS) * (hi - lo)
        self._val_label.setText(f"{val:.4g}")
        self._update_timer.start()

    def _reset_to_base(self):
        col = self._col_combo.currentData()
        if not col:
            return
        try:
            cur = float(self._base_input.get(col, 0))
        except (ValueError, TypeError):
            cur = 0.0
        lo = self._min_spin.value()
        hi = self._max_spin.value()
        if hi > lo:
            frac = max(0.0, min(1.0, (cur - lo) / (hi - lo)))
            self._slider.setValue(int(frac * _SLIDER_STEPS))

    def _current_value(self):
        lo = self._min_spin.value()
        hi = self._max_spin.value()
        return lo + (self._slider.value() / _SLIDER_STEPS) * (hi - lo)

    # ── 온도 보정 (물리 기반) ─────────────────────────────────────────────────

    @staticmethod
    def _temp_strength_factor(temp_k: float) -> float:
        """오스테나이트계 스테인리스강 강도 온도 보정 계수 (정규화).
        저온: 강도 소폭 증가 / 고온: 강도 감소"""
        T_C = float(temp_k) - 273.15
        if T_C < 20.0:
            # 저온 강화 (오스테나이트 → 마르텐사이트 변태 효과 반영)
            return float(np.clip(1.0 + (20.0 - T_C) * 0.00020, 1.0, 1.20))
        else:
            return float(np.clip(1.0 - (T_C - 20.0) * 0.00055, 0.20, 1.0))

    @staticmethod
    def _temp_ductility_factor(temp_k: float) -> float:
        """온도에 따른 연성 보정 계수. 저온: 연성 감소 / 고온: 연성 증가."""
        T_C = float(temp_k) - 273.15
        if T_C < 20.0:
            return float(np.clip(1.0 - (20.0 - T_C) * 0.00015, 0.5, 1.0))
        else:
            return float(np.clip(1.0 + (T_C - 20.0) * 0.00030, 1.0, 2.5))

    def _apply_temperature_correction(self, mean: np.ndarray, temp_k_new: float) -> np.ndarray:
        """모델이 온도 영향을 학습하지 못한 경우 물리 기반 보정을 적용한다."""
        try:
            temp_k_base = float(self._base_input.get("Temperature (K)", 293.15))
        except (ValueError, TypeError):
            temp_k_base = 293.15

        sf = self._temp_strength_factor(temp_k_new) / max(self._temp_strength_factor(temp_k_base), 1e-9)
        df = self._temp_ductility_factor(temp_k_new) / max(self._temp_ductility_factor(temp_k_base), 1e-9)

        corrected = np.array(mean, dtype=float)
        corrected[0] = max(1.0, mean[0] * sf)
        corrected[1] = max(corrected[0] + 1.0, mean[1] * sf)
        corrected[2] = float(np.clip(mean[2] * df, 2.0, 120.0))
        corrected[3] = float(np.clip(mean[3] * df, 0.0, 95.0))
        return corrected

    # ── 예측 + 렌더링 ────────────────────────────────────────────────────────

    def _compute_base_mean(self) -> np.ndarray | None:
        """다이얼로그 초기화 시점의 기준 예측값을 계산한다."""
        try:
            scaled = self._data_engine.get_inference_data(self._base_input)
            mean_s, _ = self._model_engine.predict(scaled.astype(np.float32))
            return self._data_engine.scaler_y.inverse_transform(mean_s)[0]
        except Exception:
            return None

    def _update_curve(self):
        if not self._model_engine or not self._data_engine:
            return
        col = self._col_combo.currentData()
        if not col:
            return

        val = self._current_value()
        modified = dict(self._base_input)
        modified[col] = str(val)

        temp_note = ""
        if col == "Temperature (K)":
            # 모델은 온도 영향을 학습하지 못하므로 기준 예측값에 물리 보정만 적용
            if self._base_mean is None:
                return
            mean = self._apply_temperature_correction(self._base_mean.copy(), val)
            temp_note = "<br><span style='font-size:10px;color:#94A3B8;'>* 물리 기반 온도 보정 적용</span>"
        else:
            try:
                scaled = self._data_engine.get_inference_data(modified)
                mean_s, _ = self._model_engine.predict(scaled.astype(np.float32))
                mean = self._data_engine.scaler_y.inverse_transform(mean_s)[0]
            except Exception:
                return

        self._result_label.setText(
            f"<b>예측 물성</b><br>"
            f"항복강도: <b>{mean[0]:.1f} MPa</b><br>"
            f"인장강도(UTS): <b>{mean[1]:.1f} MPa</b><br>"
            f"연신율: <b>{mean[2]:.1f} %</b><br>"
            f"단면감소율: <b>{mean[3]:.1f} %</b>"
            f"{temp_note}"
        )

        strain, stress, points, meta, segments = self._build_fn(mean, modified)
        col_label = _COLUMN_LABELS.get(col, col)
        self._render(strain, stress, points, segments, col_label, val)

    def _render(self, strain, stress, points, segments, col_label, col_val):
        self._canvas_fig.clear()
        ax = self._canvas_fig.add_subplot(111)
        ax.set_facecolor("#FAFAFA")

        ax.set_title(
            f"Stress-Strain Curve  |  {col_label} = {col_val:.4g}",
            fontsize=12, fontweight="bold", color="#111827", pad=10,
        )
        ax.set_xlabel("Strain", fontsize=11, color="#334155")
        ax.set_ylabel("Stress (MPa)", fontsize=11, color="#334155")
        ax.tick_params(colors="#64748B", labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#E2E8F0")

        yield_x  = points["Yield"][0]
        uts_x    = points["UTS"][0]
        frac_x   = points["Fracture"][0]

        ax.axvspan(0.0,     yield_x, color="#2563EB", alpha=0.06)
        ax.axvspan(yield_x, uts_x,   color="#F59E0B", alpha=0.05)
        ax.axvspan(uts_x,   frac_x,  color="#DC2626", alpha=0.04)

        seg_colors = {"elastic": "#2563EB", "hardening": "#F59E0B", "necking": "#DC2626"}
        for name, (xs, ys) in segments.items():
            ax.plot(xs, ys, color=seg_colors[name], linewidth=2.8, solid_capstyle="round")
            ax.fill_between(xs, ys, 0, color=seg_colors[name], alpha=0.06)

        pt_colors = {"Yield": "#2563EB", "UTS": "#DC2626", "Fracture": "#059669"}
        offsets   = {"Yield": (12, 12), "UTS": (-70, -42), "Fracture": (-82, -6)}
        bbox_style = {"boxstyle": "round,pad=0.22", "facecolor": "#FFFFFF",
                      "edgecolor": "#CBD5E1", "alpha": 0.92}
        for name, (xv, yv) in points.items():
            c = pt_colors[name]
            ax.scatter([xv], [yv], s=42, color=c, zorder=5)
            ax.annotate(
                f"{name}\n({xv:.3f}, {yv:.0f} MPa)",
                xy=(xv, yv), xytext=offsets[name],
                textcoords="offset points", fontsize=9,
                color=c, fontweight="bold", bbox=bbox_style,
                arrowprops={"arrowstyle": "-", "color": c, "lw": 1.0},
            )

        uts_y = points["UTS"][1]
        frac_x = points["Fracture"][0]
        ax.set_xlim(0.0, frac_x * 1.08)
        ax.set_ylim(0.0, uts_y * 1.18)
        self._canvas_fig.tight_layout(pad=0.4)
        self._canvas.draw()
