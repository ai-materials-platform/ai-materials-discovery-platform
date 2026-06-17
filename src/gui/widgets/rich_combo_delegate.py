from PyQt6.QtCore import Qt, QRect, QSize
from PyQt6.QtGui import QColor, QFont, QPainter
from PyQt6.QtWidgets import QComboBox, QStyle, QStyledItemDelegate


class WidePopupComboBox(QComboBox):
    """팝업 드롭다운을 고정 최소 너비로 펼치는 콤보박스."""
    _POPUP_MIN_WIDTH = 300

    def showPopup(self):
        self.view().setMinimumWidth(self._POPUP_MIN_WIDTH)
        super().showPopup()


class RichComboDelegate(QStyledItemDelegate):
    """Excel 테마 선택창 스타일의 콤보박스 아이템 (뱃지 + 제목 + 설명)."""

    _ITEMS = {
        "평균값으로 채우기":        ("AVG", "열 평균으로 결측치 보완"),
        "중앙값으로 채우기":        ("MED", "열 중앙값으로 결측치 보완"),
        "주변 값으로 예측(KNN)":    ("KNN", "인접 샘플 기반 보간"),
        "해당 행 제거":             ("DEL", "결측치 포함 행 삭제"),
        "감지 범위로 보정":         ("IQR", "IQR 경계로 값 클리핑"),
        "이상치 행 제거":           ("DEL", "이상치 포함 행 삭제"),
        "표시만 하고 유지":         ("TAG", "이상치 표시 후 유지"),
        "잘못된 값을 NaN으로 변환": ("NaN", "오류값 NaN으로 마킹"),
        "잘못된 값이 있는 행 제거": ("DEL", "형식 오류 포함 행 삭제"),
    }

    def __init__(self, dark_mode: bool = False, parent=None):
        super().__init__(parent)
        self.dark_mode = dark_mode

    def sizeHint(self, option, index):
        return QSize(option.rect.width() if option.rect.width() > 0 else 220, 50)

    def paint(self, painter: QPainter, option, index):
        painter.save()

        text = index.data(Qt.ItemDataRole.DisplayRole) or ""
        badge_code, desc = self._ITEMS.get(text, ("···", ""))

        if self.dark_mode:
            bg_normal   = QColor("#2F3339")
            bg_selected = QColor("#3A4048")
            badge_bg    = QColor("#404850")
            badge_fg    = QColor("#D5DBE3")
            title_fg    = QColor("#F3F4F6")
            desc_fg     = QColor("#94A3B8")
            sep_color   = QColor("#4F5965")
        else:
            bg_normal   = QColor("#FFFFFF")
            bg_selected = QColor("#F1F5F9")
            badge_bg    = QColor("#E9EEF3")
            badge_fg    = QColor("#374151")
            title_fg    = QColor("#111827")
            desc_fg     = QColor("#6B7280")
            sep_color   = QColor("#E2E8F0")

        is_selected = bool(option.state & QStyle.StateFlag.State_Selected)
        painter.fillRect(option.rect, bg_selected if is_selected else bg_normal)

        # 하단 구분선
        painter.setPen(sep_color)
        painter.drawLine(
            option.rect.left() + 8, option.rect.bottom(),
            option.rect.right() - 8, option.rect.bottom(),
        )

        # 뱃지 박스
        badge_rect = QRect(option.rect.left() + 8, option.rect.top() + 8, 40, 34)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(badge_bg)
        painter.drawRoundedRect(badge_rect, 5, 5)

        painter.setPen(badge_fg)
        badge_font = QFont("Malgun Gothic", 8)
        badge_font.setBold(True)
        painter.setFont(badge_font)
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, badge_code)

        # 제목
        text_left = option.rect.left() + 56
        title_rect = QRect(text_left, option.rect.top() + 7, option.rect.right() - text_left - 8, 18)
        painter.setPen(title_fg)
        title_font = QFont("Malgun Gothic", 10)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.drawText(title_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, text)

        # 설명
        desc_rect = QRect(text_left, option.rect.top() + 26, option.rect.right() - text_left - 8, 16)
        painter.setPen(desc_fg)
        desc_font = QFont("Malgun Gothic", 8)
        painter.setFont(desc_font)
        painter.drawText(desc_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, desc)

        painter.restore()
