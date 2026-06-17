APP_FONT_FAMILY = '"Malgun Gothic"'
APP_FONT_SIZE = 11

LIGHT_QSS = f"""
QMainWindow, QWidget {{ background-color: #F4F6F8; color: #111827; font-family: {APP_FONT_FAMILY}; font-size: {APP_FONT_SIZE}px; }}
QTabWidget::pane {{ border: 1px solid #C9D2DC; background: #FFFFFF; }}
QTabBar {{ background: #E9EEF3; }}
QTabBar::tab {{ background: #E9EEF3; color: #475569; padding: 9px 16px; border: 1px solid #C9D2DC; border-bottom: none; border-top-left-radius: 6px; border-top-right-radius: 6px; }}
QTabBar::tab:selected {{ background: #FFFFFF; color: #111827; font-weight: 700; border-bottom: 2px solid #1E293B; }}
QTabBar::tab:hover {{ background: #FFFFFF; color: #111827; }}
QGroupBox {{ border: 1px solid #C9D2DC; border-radius: 10px; margin-top: 12px; padding-top: 12px; font-weight: 700; color: #28323C; background: #FFFFFF; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #475569; letter-spacing: 0.3px; }}
QPushButton {{ background-color: #EEF2F6; color: #111827; border: 1px solid #C9D2DC; border-radius: 8px; padding: 7px 14px; }}
QPushButton:hover {{ background-color: #E2E8F0; }}
QPushButton:disabled {{ color: #7C8794; background: #F4F5F6; border-color: #E2E7EC; }}
QComboBox {{ border: 1px solid #C9D2DC; border-radius: 8px; background: #FFFFFF; padding: 6px 10px; color: #111827; }}
QComboBox:focus {{ border-color: #1E293B; }}
QComboBox QAbstractItemView {{ background: #FFFFFF; border: 1px solid #C9D2DC; color: #111827; selection-background-color: #1E293B; selection-color: #FFFFFF; }}
QTableWidget {{ border: 1px solid #C9D2DC; gridline-color: #E2E8F0; background: #FFFFFF; alternate-background-color: #F8FAFC; color: #111827; }}
QTableWidget::item {{ background-color: #FFFFFF; color: #111827; }}
QTableWidget::item:alternate {{ background-color: #F8FAFC; }}
QTableWidget::item:selected {{ background: #1E293B; color: #FFFFFF; }}
QHeaderView::section {{ background: #E9EEF3; color: #475569; border: none; border-right: 1px solid #C9D2DC; border-bottom: 1px solid #C9D2DC; padding: 7px 10px; font-weight: 700; }}
QScrollBar:vertical {{ width: 8px; background: transparent; }}
QScrollBar::handle:vertical {{ background: #B8C2CE; border-radius: 4px; min-height: 24px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ height: 8px; background: transparent; }}
QScrollBar::handle:horizontal {{ background: #B8C2CE; border-radius: 4px; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
QLineEdit {{ border: 1px solid #C9D2DC; border-radius: 8px; background: #FFFFFF; padding: 7px 10px; color: #111827; selection-background-color: #1E293B; }}
QLineEdit:focus {{ border-color: #1E293B; }}
QDoubleSpinBox, QSpinBox {{ border: 1px solid #C9D2DC; border-radius: 8px; background: #FFFFFF; padding: 7px 10px; color: #111827; }}
QSplitter::handle {{ background: #C9D2DC; }}
QDialog {{ background: #FFFFFF; }}
QMessageBox {{ background: #FFFFFF; }}
QToolTip {{ background: #FFFFFF; color: #1E293B; border: 1px solid #CBD5E1; border-radius: 6px; padding: 6px 10px; font-size: 11px; }}
"""

DARK_QSS = f"""
QMainWindow, QWidget {{ background-color: #25282D; color: #F3F4F6; font-family: {APP_FONT_FAMILY}; font-size: {APP_FONT_SIZE}px; }}
QTabWidget {{ background: #25282D; }}
QTabWidget::pane {{ border: 1px solid #4F5965; background: #2F3339; }}
QTabBar {{ background: #25282D; }}
QTabBar::scroller {{ background: #25282D; }}
QTabBar QToolButton {{ background: #25282D; border: none; }}
QTabBar::tab {{ background: #25282D; color: #B6C0CB; padding: 9px 16px; border: 1px solid #4F5965; border-bottom: none; border-top-left-radius: 6px; border-top-right-radius: 6px; }}
QTabBar::tab:selected {{ background: #2F3339; color: #F8FAFC; font-weight: 700; border-bottom: 2px solid #1E293B; }}
QTabBar::tab:hover {{ background: #333840; color: #F3F4F6; }}
QGroupBox {{ border: 1px solid #4F5965; border-radius: 10px; margin-top: 12px; padding-top: 12px; font-weight: 700; color: #E2E8F0; background: #2F3339; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #D5DBE3; letter-spacing: 0.3px; }}
QPushButton {{ background-color: #3A4048; color: #F3F4F6; border: 1px solid #59616C; border-radius: 8px; padding: 7px 14px; }}
QPushButton:hover {{ background-color: #454C55; color: #FFFFFF; }}
QPushButton:disabled {{ color: #7D8794; background: #2F3339; border-color: #3F454D; }}
QComboBox {{ border: 1px solid #59616C; border-radius: 8px; background: #2F3339; padding: 6px 10px; color: #F3F4F6; }}
QComboBox:focus {{ border-color: #1E293B; }}
QComboBox QAbstractItemView {{ background: #2F3339; border: 1px solid #59616C; color: #F3F4F6; selection-background-color: #1E293B; selection-color: #FFFFFF; }}
QTableWidget {{ border: 1px solid #4F5965; gridline-color: #3A4048; background: #2F3339; alternate-background-color: #25282D; color: #F3F4F6; }}
QTableWidget::item {{ background-color: #2F3339; color: #F3F4F6; }}
QTableWidget::item:alternate {{ background-color: #25282D; }}
QTableWidget::item:selected {{ background: #1E293B; color: #FFFFFF; }}
QHeaderView::section {{ background: #25282D; color: #D5DBE3; border: none; border-right: 1px solid #4F5965; border-bottom: 1px solid #4F5965; padding: 7px 10px; font-weight: 700; }}
QScrollBar:vertical {{ width: 8px; background: transparent; }}
QScrollBar::handle:vertical {{ background: #59616C; border-radius: 4px; min-height: 24px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ height: 8px; background: transparent; }}
QScrollBar::handle:horizontal {{ background: #59616C; border-radius: 4px; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
QLineEdit {{ border: 1px solid #59616C; border-radius: 8px; background: #2F3339; padding: 7px 10px; color: #F3F4F6; selection-background-color: #1E293B; }}
QLineEdit:focus {{ border-color: #1E293B; }}
QDoubleSpinBox, QSpinBox {{ border: 1px solid #59616C; border-radius: 8px; background: #2F3339; padding: 7px 10px; color: #F3F4F6; }}
QSplitter::handle {{ background: #4F5965; }}
QDialog {{ background: #2F3339; color: #F3F4F6; }}
QMessageBox {{ background: #2F3339; color: #F3F4F6; }}
QScrollArea {{ background: transparent; border: none; }}
QToolTip {{ background: #2F3339; color: #F8FAFC; border: 1px solid #4F5965; border-radius: 6px; padding: 6px 10px; font-size: 11px; }}
"""

GLOBAL_QSS = LIGHT_QSS

CURVE_SEGMENT_STYLES = {
    "elastic": {
        "color": "#2563EB",
        "fill": "#93C5FD",
        "label": "Elastic region",
        "legend_html": '<span style="color:#2563EB; font-weight:700;">파란색</span> = Elastic region',
    },
    "hardening": {
        "color": "#F59E0B",
        "fill": "#FCD34D",
        "label": "Plastic hardening",
        "legend_html": '<span style="color:#F59E0B; font-weight:700;">주황색</span> = Plastic hardening',
    },
    "necking": {
        "color": "#DC2626",
        "fill": "#FCA5A5",
        "label": "Necking",
        "legend_html": '<span style="color:#DC2626; font-weight:700;">빨간색</span> = Necking',
    },
}
