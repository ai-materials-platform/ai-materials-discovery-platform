import os

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

# ── 오스테나이트계 스테인리스강 전문 시스템 프롬프트 ─────────────────────────
_SYSTEM_PROMPT = """
당신은 오스테나이트계 스테인리스강 전문 재료공학자이자 이 AI 예측 플랫폼의 전문 보조 시스템입니다.

## 이 플랫폼 정보
오스테나이트계 스테인리스강의 4가지 기계적 물성을 화학 조성·공정 조건으로 예측합니다.
- 예측 물성: 0.2% 항복강도(MPa), 인장강도 UTS(MPa), 연신율(%), 단면감소율(%)
- 앙상블 모델: RF(Random Forest), GBM(Gradient Boosting), MLP(신경망), TFP(불확실성 정량화)
- 엔지니어드 피처: Ni_eq, Cr_eq, Cr/Ni 비율, C+N

## 합금 성분별 역할
| 원소 | 범위(wt%) | 주요 역할 |
|------|-----------|-----------|
| Cr | 16~26 | 내식성 핵심. Cr₂O₃ 부동태막 형성. Cr_eq = Cr + Mo + 1.5Si + 0.5Nb |
| Ni | 6~22 | 오스테나이트 안정화, 연성·인성 향상. Ni_eq = Ni + 30C + 0.5Mn + 30N + 0.3Cu |
| Mo | 0~3 | 공식(pitting) 저항성, 고온 크리프 강도 향상 |
| N | 0~0.5 | 강력한 고용강화 (C의 ~30배). 입계 예민화 억제 |
| C | 0~0.15 | 고용강화, 과다 시 Cr₂₃C₆ 석출 → 예민화(sensitization) 위험 |
| Mn | 0~2 | 오스테나이트 안정화, N 고용도 증가 |
| Si | 0~1 | 산화 저항성, 내산성 향상 |
| Nb, Ti | 0~0.5 | 탄질화물 형성으로 예민화 억제 (안정화강) |
| Cu | 0~3 | 내산성 향상, 가공경화율 감소 |

## 주요 강종 특성
- AISI 304 (18Cr-8Ni): 범용, YS 205~310 MPa, UTS 515~620 MPa, El 40~60%
- AISI 316 (16Cr-10Ni-2Mo): 내식성↑, YS 210~310 MPa, UTS 515~620 MPa
- AISI 310 (25Cr-20Ni): 고온 내열용, YS 205 MPa, UTS 515 MPa
- AISI 321 (Ti 안정화): 예민화 저항, YS 205 MPa, UTS 515 MPa, El 40%
- AISI 347 (Nb 안정화): 용접부 내식성 우수

## 기계적 물성 결정 인자
**항복강도 (YS):**
- 고용강화: ΔYS ≈ 470·[N] + 37·[C] + 9·[Cr] + 4·[Ni] (MPa, wt% 단위)
- 결정립 미세화: Hall-Petch σ_y = σ₀ + k·d^(-1/2)
- 가공경화: 오스테나이트 → ε·α' 마르텐사이트 변태

**연신율 (El):**
- Ni, N 증가 → El 증가 (오스테나이트 안정화)
- 탄소량 감소 → El 증가
- 결정립 조대화 → El 증가 (단, YS 감소)

**단면감소율 (RA):**
- 내부 결함(개재물, 기공) 에 민감
- 고용화 처리 완전도에 크게 의존

## 열처리 공정
- **고용화 처리 (Solution Annealing):** 1010~1150°C → 급냉(수냉/공냉)
  → 오스테나이트 단상화, 탄화물 완전 재용해
  → 온도 낮을수록 결정립 미세 → YS↑, El↓
- **예민화 (Sensitization):** 425~870°C 장시간 노출
  → Cr₂₃C₆ 입계 석출 → 입계 부식 위험
  → 저탄소(L급) 또는 안정화강으로 방지
- **응력제거 풀림:** 850~950°C → 내부 응력 완화

## Schaeffler 선도 활용
- Cr_eq = Cr + Mo + 1.5Si + 0.5Nb
- Ni_eq = Ni + 30C + 0.5Mn + 30N + 0.3Cu
- 순 오스테나이트 구역: Ni_eq > 9 + 0.6·Cr_eq (근사)
- δ-페라이트 혼입 시: 고온 균열 저항성↑, 자성↑, 인성↓

## 답변 원칙
1. **반드시 한국어(한글)로만 답변**하세요. 일본어·중국어·한자는 절대 사용 금지입니다.
2. 전문 용어는 영어 원어를 유지하고 (예: austenite, yield strength), 나머지는 모두 한글로 작성하세요.
3. 수치·수식 근거 포함 (MPa, wt% 등 단위 명기)
4. 예측 결과가 제공되면 해당 합금 조성을 구체적으로 분석
5. 실용적 조언 우선 (설계 개선안, 공정 조정 방향 제시)
6. 불확실한 내용은 불확실하다고 명시
""".strip()


_OPENAI_URL = "https://api.openai.com/v1/chat/completions"
_OPENAI_MODEL_DEFAULT = "gpt-4o-mini"


def _strip_cjk(text: str) -> str:
    """한자·일본어(히라가나·가타카나) 문자를 제거한다. 한글은 유지."""
    import re  # noqa: PLC0415
    # CJK Unified Ideographs, Extension A/B, Radicals
    # Hiragana, Katakana, Bopomofo 등 제거
    return re.sub(
        r"[　-〿぀-ヿ㄀-ㄯ"
        r"㈀-㋿㐀-䶿一-鿿"
        r"豈-﫿︰-﹏\U00020000-\U0002a6df]+",
        "",
        text,
    )


class _OpenAIWorker(QThread):
    response_received = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, api_key: str, model: str, messages: list, system_prompt: str):
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.messages = messages
        self.system_prompt = system_prompt

    def run(self):
        try:
            import requests  # noqa: PLC0415

            payload = {
                "model": self.model,
                "messages": [{"role": "system", "content": self.system_prompt}] + self.messages,
                "max_tokens": 1024,
                "temperature": 0.7,
            }

            resp = requests.post(
                _OPENAI_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=60,
            )

            data = resp.json()

            if resp.status_code != 200 or "error" in data:
                err_obj = data.get("error", {})
                err_msg = err_obj.get("message", "") or str(err_obj) or resp.text
                err_code = err_obj.get("code", resp.status_code)
                self.error_occurred.emit(f"OpenAI 오류 [{err_code}]: {err_msg}")
                return

            text = data["choices"][0]["message"]["content"]
            self.response_received.emit(text)

        except Exception as exc:  # noqa: BLE001
            self.error_occurred.emit(f"오류: {exc}")


class LLMChatMixin:
    """Google Gemini 기반 오스테나이트계 스테인리스강 전문 AI 채팅 탭."""

    def _create_llm_chat_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        # ── 상단 옵션 바 ──────────────────────────────────────────────────────
        option_row = QHBoxLayout()

        self._llm_include_ctx_cb = QCheckBox("현재 예측 결과 포함")
        self._llm_include_ctx_cb.setChecked(True)
        self._llm_include_ctx_cb.setStyleSheet("font-size: 11px;")
        option_row.addWidget(self._llm_include_ctx_cb)
        option_row.addStretch()

        clear_btn = QPushButton("대화 초기화")
        clear_btn.setFixedHeight(28)
        clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        clear_btn.setStyleSheet(
            "QPushButton { background: #F1F5F9; color: #475569; border: 1px solid #CBD5E1; "
            "border-radius: 6px; font-size: 11px; font-weight: 600; padding: 0 12px; }"
            "QPushButton:hover { background: #E2E8F0; }"
        )
        clear_btn.clicked.connect(self._llm_clear_chat)
        option_row.addWidget(clear_btn)
        layout.addLayout(option_row)

        # ── 채팅 히스토리 스크롤 영역 ─────────────────────────────────────────
        self._llm_scroll = QScrollArea()
        self._llm_scroll.setWidgetResizable(True)
        self._llm_scroll.setFrameShape(QFrame.Shape.NoFrame)

        self._llm_chat_container = QWidget()
        self._llm_chat_layout = QVBoxLayout(self._llm_chat_container)
        self._llm_chat_layout.setContentsMargins(2, 4, 2, 4)
        self._llm_chat_layout.setSpacing(10)
        self._llm_chat_layout.addStretch()

        self._llm_scroll.setWidget(self._llm_chat_container)
        layout.addWidget(self._llm_scroll, 1)

        # 웰컴 메시지
        self._llm_add_bubble(
            "안녕하세요! 오스테나이트계 스테인리스강 전문 AI입니다.\n"
            "합금 조성 분석·물성 해석·열처리 공정·설계 개선에 대해 질문해 주세요.\n"
            "'현재 예측 결과 포함'을 체크하면 예측값을 바탕으로 구체적으로 답변합니다.",
            is_user=False,
        )

        # ── 메시지 입력 바 ────────────────────────────────────────────────────
        input_row = QHBoxLayout()

        self._llm_input = QLineEdit()
        self._llm_input.setPlaceholderText("질문을 입력하세요 (예: 이 합금의 항복강도를 높이려면?)")
        self._llm_input.setFixedHeight(36)
        self._llm_input.returnPressed.connect(self._llm_send)
        input_row.addWidget(self._llm_input, 1)

        self._llm_send_btn = QPushButton("전송")
        self._llm_send_btn.setFixedSize(60, 36)
        self._llm_send_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._llm_send_btn.setStyleSheet(
            "QPushButton { background: #E56020; color: white; border: none; "
            "border-radius: 8px; font-weight: 700; font-size: 12px; }"
            "QPushButton:hover { background: #C94D10; }"
            "QPushButton:disabled { background: #94A3B8; }"
        )
        self._llm_send_btn.clicked.connect(self._llm_send)
        input_row.addWidget(self._llm_send_btn)
        layout.addLayout(input_row)

        # ── 모델명 입력 ───────────────────────────────────────────────────────
        model_row = QHBoxLayout()

        model_lbl = QLabel("모델:")
        model_lbl.setFixedWidth(48)
        model_lbl.setStyleSheet("font-size: 11px; color: #64748B;")
        model_row.addWidget(model_lbl)

        self._llm_model_edit = QLineEdit()
        self._llm_model_edit.setFixedHeight(30)
        self._llm_model_edit.setPlaceholderText(
            "OpenAI 모델명 (예: gpt-4o-mini, gpt-4o, gpt-3.5-turbo)"
        )
        self._llm_model_edit.setText(_OPENAI_MODEL_DEFAULT)

        # API 키 위젯 (숨김 — .env 파일에서 자동 로드)
        self._llm_api_key_edit = QLineEdit()
        self._llm_api_key_edit.setText(os.environ.get("OPENAI_API_KEY", ""))

        # 내부 상태
        self._llm_messages: list[dict] = []
        self._llm_worker: _GeminiWorker | None = None

        return tab

    # ── 메시지 전송 ────────────────────────────────────────────────────────────

    def _llm_send(self):
        text = self._llm_input.text().strip()
        if not text:
            return

        api_key = (
            self._llm_api_key_edit.text().strip()
            or os.environ.get("GEMINI_API_KEY", "")
        )
        if not api_key:
            self._llm_add_bubble(
                "API 키를 입력해 주세요.\n"
                "platform.openai.com → API Keys → Create new secret key",
                is_user=False,
            )
            return

        if not api_key.startswith("sk-") or len(api_key) < 20:
            self._llm_add_bubble(
                "API 키 형식이 올바르지 않습니다.\n"
                "OpenAI 키는 'sk-' 로 시작합니다.\n"
                "키 입력란을 지우고 platform.openai.com 에서 발급받은 키를 다시 입력해 주세요.",
                is_user=False,
            )
            self._llm_api_key_edit.clear()
            return

        self._llm_input.clear()
        self._llm_add_bubble(text, is_user=True)
        self._llm_send_btn.setEnabled(False)
        self._llm_send_btn.setText("…")

        # 예측 결과 컨텍스트를 시스템 프롬프트에 추가
        system_prompt = _SYSTEM_PROMPT
        if self._llm_include_ctx_cb.isChecked():
            ctx = self._llm_build_context()
            if ctx:
                system_prompt += f"\n\n--- 현재 예측 결과 (이 합금을 분석에 활용하세요) ---\n{ctx}"

        self._llm_messages.append({"role": "user", "content": text})

        model = self._llm_model_edit.text().strip() or _OPENAI_MODEL_DEFAULT
        self._llm_worker = _OpenAIWorker(api_key, model, list(self._llm_messages), system_prompt)
        self._llm_worker.response_received.connect(self._llm_on_response)
        self._llm_worker.error_occurred.connect(self._llm_on_error)
        self._llm_worker.start()

    def _llm_on_response(self, text: str):
        self._llm_messages.append({"role": "assistant", "content": text})
        self._llm_add_bubble(text, is_user=False)
        self._llm_send_btn.setEnabled(True)
        self._llm_send_btn.setText("전송")

    def _llm_on_error(self, text: str):
        self._llm_add_bubble(f"[오류] {text}", is_user=False)
        self._llm_send_btn.setEnabled(True)
        self._llm_send_btn.setText("전송")

    def _llm_clear_chat(self):
        self._llm_messages = []
        while self._llm_chat_layout.count() > 1:
            item = self._llm_chat_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._llm_add_bubble("대화가 초기화되었습니다. 질문을 입력해 주세요.", is_user=False)

    # ── 버블 생성 ──────────────────────────────────────────────────────────────

    def _llm_add_bubble(self, text: str, *, is_user: bool):
        from PyQt6.QtCore import QTimer  # noqa: PLC0415

        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lbl.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        lbl.setMaximumWidth(520)

        if is_user:
            lbl.setStyleSheet(
                "QLabel { background: #E56020; color: white; border-radius: 12px; "
                "padding: 10px 14px; font-size: 12px; }"
            )
        else:
            lbl.setStyleSheet(
                "QLabel { background: #F1F5F9; color: #1E293B; border-radius: 12px; "
                "padding: 10px 14px; font-size: 12px; border: 1px solid #E2E8F0; }"
            )

        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)

        if is_user:
            row.addStretch()
            row.addWidget(lbl)
        else:
            row.addWidget(lbl)
            row.addStretch()

        self._llm_chat_layout.insertWidget(self._llm_chat_layout.count() - 1, container)

        QTimer.singleShot(
            60,
            lambda: self._llm_scroll.verticalScrollBar().setValue(
                self._llm_scroll.verticalScrollBar().maximum()
            ),
        )

    # ── 예측 컨텍스트 수집 ────────────────────────────────────────────────────

    def _llm_build_context(self) -> str:
        PROP_NAMES = ["항복강도(MPa)", "UTS(MPa)", "연신율(%)", "단면감소율(%)"]

        for attr, label in (
            ("_pretrained_prediction_state", "사전학습 모델"),
            ("_user_prediction_state", "사용자 학습 모델"),
        ):
            state = getattr(self, attr, None)
            if state is None:
                continue
            mean = state.get("mean")
            std = state.get("std")
            inp = state.get("input_dict", {})
            if mean is None:
                continue
            lines = [f"[{label} 예측 결과]"]
            for name, m, s in zip(PROP_NAMES, mean, std):
                lines.append(f"  {name}: {float(m):.1f} ± {float(s):.1f}")
            if inp:
                lines.append("[입력 조성 / 공정 (wt% 또는 조건값)]")
                lines.extend(f"  {k}: {v}" for k, v in list(inp.items())[:16])
            return "\n".join(lines)

        return ""

    # ── 다크모드 테마 ──────────────────────────────────────────────────────────

    def _apply_llm_chat_theme(self):
        if not hasattr(self, "_llm_input"):
            return
        c = self._theme()
        input_style = (
            f"QLineEdit {{ background: {c['input_bg']}; color: {c['text_primary']}; "
            f"border: 1px solid {c['border']}; border-radius: 8px; padding: 6px 10px; }}"
            "QLineEdit:focus { border-color: #E56020; }"
        )
        self._llm_input.setStyleSheet(input_style)

        ai_bg = "#2D3748" if self._dark_mode else "#F1F5F9"
        ai_text = "#E2E8F0" if self._dark_mode else "#1E293B"
        ai_border = "#4A5568" if self._dark_mode else "#E2E8F0"
        ai_style = (
            f"QLabel {{ background: {ai_bg}; color: {ai_text}; border-radius: 12px; "
            f"padding: 10px 14px; font-size: 12px; border: 1px solid {ai_border}; }}"
        )

        for i in range(self._llm_chat_layout.count() - 1):
            item = self._llm_chat_layout.itemAt(i)
            if not item or not item.widget():
                continue
            row_layout = item.widget().layout()
            if not row_layout:
                continue
            for j in range(row_layout.count()):
                child = row_layout.itemAt(j)
                if child and child.widget() and isinstance(child.widget(), QLabel):
                    lbl = child.widget()
                    if "#E56020" in lbl.styleSheet():
                        continue
                    lbl.setStyleSheet(ai_style)
