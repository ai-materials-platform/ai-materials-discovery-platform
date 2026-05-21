import matplotlib.pyplot as plt
from PyQt6.QtWidgets import QApplication, QMainWindow

from src.engine.data_engine import DataEngine
from src.gui.constants import APP_FONT_SIZE
from src.gui.mixins import (
    ChartsMixin,
    FeatureSelectionMixin,
    InferenceMixin,
    PreprocessingMixin,
    SettingsPanelMixin,
    ThemeMixin,
    TrainingMixin,
    UISetupMixin,
    WorkspaceMixin,
)

try:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass


class MainWindow(
    QMainWindow,
    UISetupMixin,
    ThemeMixin,
    SettingsPanelMixin,
    PreprocessingMixin,
    FeatureSelectionMixin,
    TrainingMixin,
    InferenceMixin,
    ChartsMixin,
    WorkspaceMixin,
):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Materials Discovery Platform")
        screen = QApplication.primaryScreen().availableGeometry()
        init_w = min(1400, max(960, int(screen.width() * 0.88)))
        init_h = min(900, max(680, int(screen.height() * 0.88)))
        self.resize(init_w, init_h)
        self.setMinimumSize(900, 650)

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
        self._panel_widgets = []
        self._panel_header_widgets = []
        self._divider_widgets = []
        self._info_box_widgets = []
        self._section_lbl_widgets = []
        self._muted_bg_widgets = []
        self._prediction_input_groups = []
        self._prediction_input_fields = []
        self._prediction_input_labels = []
        self._curve_info_panels = []
        self._curve_legend_cards = []
        self._curve_legend_label_widgets = []
        self._simulation_status_labels = []
        self._simulation_detail_labels = []
        self._simulation_assumption_labels = []
        self._simulation_control_cards = []
        self._simulation_reset_buttons = []
        self._simulation_widgets = []
        self._pretrained_prediction_state = None
        self._user_prediction_state = None

        self.init_ui()
