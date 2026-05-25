from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication, QDoubleSpinBox, QLabel

from src.gui.constants import DARK_QSS, LIGHT_QSS


class ThemeMixin:
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
        QApplication.instance().setPalette(palette)

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

        for w in self._panel_widgets:
            w.setStyleSheet(f"background: {c['panel_bg']};")

        for w in self._panel_header_widgets:
            w.setStyleSheet(
                f"font-size: 11px; color: {c['text_label']}; padding: 10px 14px; "
                f"letter-spacing: 0.8px; font-weight: 600; border-bottom: 1px solid {c['divider']};"
            )

        for w in self._divider_widgets:
            w.setStyleSheet(f"background: {c['divider']};")

        for w in self._info_box_widgets:
            w.setStyleSheet(
                f"font-size: 12px; color: {c['text_sec']}; background: {c['info_bg']}; "
                f"padding: 10px; border-left: 3px solid {c['accent']}; border-radius: 6px;"
            )

        for w in self._section_lbl_widgets:
            w.setStyleSheet(
                f"font-size: 11px; color: {c['text_label']}; "
                "font-weight: 600; letter-spacing: 0.4px;"
            )

        for w in self._muted_bg_widgets:
            w.setStyleSheet(
                f"font-size: 12px; color: {c['text_sec']}; padding: 10px; "
                f"background: {c['muted_bg']}; border: 1px solid {c['border']}; border-radius: 6px;"
            )

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
        for attr_name in ["result_display", "pretrained_result_display"]:
            if hasattr(self, attr_name):
                getattr(self, attr_name).setStyleSheet(
                    f"font-size: 12px; font-weight: 600; color: {c['text_primary']}; "
                    f"background: {c['muted_bg']}; border: 1px solid {c['border']}; "
                    "border-radius: 10px; padding: 12px;"
                )
        for panel in self._curve_info_panels:
            panel.setStyleSheet(
                f"background: {c['muted_bg']}; border: 1px solid {c['border']}; border-radius: 10px;"
            )
        for attr_name in ["pretrained_curve_placeholder", "stress_strain_placeholder_label"]:
            if hasattr(self, attr_name):
                getattr(self, attr_name).setStyleSheet(
                    f"font-size: 13px; font-weight: 600; color: {c['text_sec']}; "
                    "background: transparent; border: none; padding: 4px 0 0 0;"
                )
        for label in self._simulation_detail_labels:
            label.setStyleSheet(
                f"font-size: 12px; color: {c['text_sec']}; "
                "background: transparent; border: none; padding: 2px 8px 0 8px;"
            )
        for label in self._simulation_assumption_labels:
            label.setStyleSheet(
                f"font-size: 11px; color: {c['text_label']}; "
                "background: transparent; border: none; padding: 0 8px 2px 8px;"
            )
        for card in self._simulation_control_cards:
            card.setStyleSheet(
                f"background: {c['muted_bg']}; border: 1px solid {c['border']}; border-radius: 10px;"
            )
            for label in card.findChildren(QLabel):
                if label.property("simulation_role") == "caption":
                    label.setStyleSheet(
                        f"font-size: 11px; font-weight: 700; color: {c['text_sec']}; "
                        "background: transparent; border: none; padding: 0 0 2px 0;"
                    )
                else:
                    label.setStyleSheet(
                        f"font-size: 11px; font-weight: 600; color: {c['text_sec']}; "
                        f"background: {c['panel_bg']}; border: 1px solid {c['border']}; "
                        "border-radius: 8px; padding: 7px 10px; min-height: 20px;"
                    )
            for spin_box in card.findChildren(QDoubleSpinBox):
                spin_box.setStyleSheet(
                    "QDoubleSpinBox { "
                    f"background: {c['input_bg']}; color: {c['text_primary']}; border: 1px solid {c['border']}; "
                    "border-radius: 8px; padding: 7px 10px; min-height: 20px; }"
                    "QDoubleSpinBox:focus { border-color: #E56020; }"
                )
        for button in self._simulation_reset_buttons:
            button.setStyleSheet(
                "QPushButton { background: #E56020; color: white; border: none; border-radius: 9px; "
                "font-weight: 700; padding: 8px 14px; }"
                "QPushButton:hover { background: #F97316; }"
            )
        for widget in self._simulation_widgets:
            widget.set_theme(c, self._dark_mode)
        legend_bg = "#FFFFFF" if not self._dark_mode else "#1F2937"
        legend_border = "#CBD5E1" if not self._dark_mode else "#64748B"
        legend_text = "#334155" if not self._dark_mode else "#E5E7EB"
        for card in self._curve_legend_cards:
            card.setStyleSheet(
                f"background: {legend_bg}; border: 1px solid {legend_border}; border-radius: 8px;"
            )
        for label in self._curve_legend_label_widgets:
            label.setStyleSheet(
                f"font-size: 11px; font-weight: 600; color: {legend_text}; "
                "background: transparent; border: none;"
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
        if hasattr(self, "material_prediction_page"):
            self.material_prediction_page.setStyleSheet(f"background: {c['app_bg']};")
        if hasattr(self, "user_page"):
            self.user_page.setStyleSheet(f"background: {c['app_bg']};")
        if hasattr(self, "main_mode_stack"):
            self.main_mode_stack.setStyleSheet(f"background: {c['app_bg']};")
        if hasattr(self, "_mp_prediction_guide_corner"):
            tab_bg = "#25282D" if self._dark_mode else "#E9EEF3"
            self._mp_prediction_guide_corner.setStyleSheet(
                f"background: {tab_bg}; border: none;"
            )
        if hasattr(self, "_mp_prediction_guide_btn"):
            self._mp_prediction_guide_btn.setStyleSheet(
                "QPushButton { background: #EEF6FF; color: #1D4ED8; border: 1px solid #BFDBFE; "
                "border-radius: 14px; font-size: 11px; font-weight: 700; padding: 0 14px; }"
                "QPushButton:hover { background: #DBEAFE; }"
            )
        if hasattr(self, "inference_left_frame"):
            self.inference_left_frame.setStyleSheet(
                f"background: {c['panel_bg']}; border: 1px solid {c['border']}; border-radius: 12px;"
            )
        if hasattr(self, "inference_right_frame"):
            self.inference_right_frame.setStyleSheet(
                f"background: {c['panel_bg']}; border: 1px solid {c['border']}; border-radius: 12px;"
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
        self._refresh_prediction_views_for_theme()
