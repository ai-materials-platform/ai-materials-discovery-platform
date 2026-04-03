# AI Materials Discovery Platform v2.0

> 스테인리스강(오스테나이트계)의 화학 조성과 공정 변수를 기반으로 기계적 물성을 예측하는 AI 플랫폼

---

## 개요

본 플랫폼은 재료공학 도메인 지식과 머신러닝을 결합하여, 연구자가 Excel 데이터 하나로 **전처리 → 학습 → 성능 분석 → 물성 추론**을 하나의 데스크톱 앱에서 수행할 수 있도록 설계된 시스템입니다.

### 예측 대상 물성

| 구분 | 물성 | 단위 |
|------|------|------|
| 강도 | 0.2% 항복강도 (Proof Stress) | MPa |
| 강도 | 인장강도 (UTS) | MPa |
| 연성 | 연신율 (Elongation) | % |
| 연성 | 단면 수축률 (Area Reduction) | % |

---

## 시스템 아키텍처

```
사용자
  └→ Electron 데스크톱 앱 (HTML/CSS/JS)
       └→ HTTP (localhost:5001)
            └→ Flask API Server (Python)
                 ├→ DataEngine  (전처리 / Feature Engineering)
                 └→ ModelEngine (RF / GBM / MLP / TFP 앙상블)
```

- **Electron**이 앱 실행 시 Python Flask 서버를 자식 프로세스로 기동
- Flask 준비 완료까지 **스플래시 스크린** 표시
- 이후 메인 UI와 Flask API 간 REST + SSE 통신

---

## 기술 스택

### Python 백엔드

| 라이브러리 | 용도 |
|-----------|------|
| Flask + flask-cors | REST API 서버 (포트 5001) |
| scikit-learn | RF, GBM, MLP 모델 / StandardScaler / KNNImputer |
| TensorFlow | TFP 앙상블 기반 구조 |
| TensorFlow Probability | 확률론적 예측, 불확실성 정량화 |
| pandas / NumPy | 데이터 처리, 배열 연산 |
| xlrd / openpyxl | `.xls` 및 `.xlsx` / `.xlsm` Excel 파일 파싱 |
| joblib | 모델 직렬화 (.pkl) |

### Electron 프론트엔드

| 기술 | 용도 |
|------|------|
| Electron 30+ | 크로스플랫폼 데스크톱 프레임워크 |
| HTML / CSS / JS | 메인 UI 및 스플래시 스크린 |
| Chart.js / Plotly.js | Parity Plot, 히트맵, 예측 그래프 |
| EventSource (SSE) | 학습 진행 상태 실시간 수신 |
| electron-builder | .exe / .dmg 배포 파일 생성 |

### 패키징

| 도구 | 결과물 |
|------|--------|
| PyInstaller | Python 백엔드 단일 실행파일 |
| electron-builder | macOS `.dmg` / Windows `.exe` 인스톨러 |

---

## 주요 기능

### 1. 스플래시 스크린
앱 실행 시 Word/Excel 수준의 로딩 창을 표시하여 Python 엔진 기동 시간을 자연스럽게 처리합니다.
- 단계별 상태 텍스트 (`API 서버 시작 중...` → `데이터 엔진 준비 중...`)
- 실시간 프로그레스 바
- 준비 완료 시 페이드 아웃 → 메인 창 페이드 인

### 2. 데이터 전처리

**1차 전처리 (데이터 정제)**
- 수치형 변환 및 형식 오류 처리
- 결측치 처리: 평균 / 중앙값 / KNN / 행 제거
- 도메인 기준 검증 (SSINA 오스테나이트 조성표 기반)
- 이상치 처리: IQR 기반 / 도메인 기준 / clip / 제거

**2차 전처리 (Feature Engineering)**

| 파생 변수 | 계산식 |
|----------|--------|
| `Cr_Ni_ratio` | `Cr / Ni` |
| `C_plus_N` | `C + N` |
| `Ni_eq` | `Ni + 30×C + 0.5×Mn` |
| `Cr_eq` | `Cr + Mo + 1.5×Si + 0.5×Nb` |

### 3. 피처 컬럼 선택

전처리 완료 후 학습 전에 사용할 입력 피처를 개별 선택/해제할 수 있습니다.
- 원본 변수(29개) / 파생 변수(4개) 구분하여 체크박스 목록 표시
- 선택 상태는 모델과 함께 저장되어 추론 시 자동 적용
- 최소 1개 이상 선택해야 학습 가능

### 4. 모델 학습

| 모델 | 구현 | 불확실성 산출 |
|------|------|--------------|
| Random Forest (RF) | `RandomForestRegressor` | 트리별 표준편차 |
| Gradient Boosting (GBM) | `MultiOutputRegressor(GBM)` | 휴리스틱 |
| MLP Neural Network | `MLPRegressor` + Early Stopping | 휴리스틱 |
| TFP 앙상블 | Bootstrap 5-MLP Ensemble | 앙상블 표준편차 |

- 학습 진행 상태를 SSE(Server-Sent Events)로 실시간 전달
- 사용자 정의 반복 횟수 및 조기 종료(Early Stopping) 지원

### 5. 성능 분석
- **Parity Plot**: 실제값 vs 예측값 분포 (대각선 기준선 포함)
- **상관관계 히트맵**: 반응형 (창 크기에 따라 수치 표시 자동 조절)
- **이중 Y축 그래프**: 단위가 다른 강도(MPa)와 연성(%) 동시 비교

### 6. 물성 추론
화학 조성 및 공정 조건 입력 → 4개 물성 + 불확실성(표준편차) 즉시 예측

---

## 디렉토리 구조

```
ai-materials-discovery-platform/
├── python/                       # Python 백엔드
│   ├── main_api.py               # Flask 진입점 (포트 5001)
│   ├── config/
│   │   └── settings.py           # 상수 / 경로 / 도메인 기준 중앙 관리
│   ├── engine/
│   │   ├── data_engine.py        # 전처리 엔진
│   │   └── model_engine.py       # 모델 엔진
│   └── api/
│       ├── routes_preprocess.py  # 전처리 API
│       ├── routes_train.py       # 학습 API (SSE)
│       ├── routes_performance.py # 성능 분석 API
│       └── routes_inference.py   # 추론 API
│
├── electron/                     # Electron 프론트엔드
│   ├── main.js                   # 메인 프로세스 (Python 기동 / 스플래시 제어)
│   ├── preload.js                # IPC 보안 브리지
│   ├── splash.html               # 스플래시 스크린
│   ├── assets/                   # 아이콘 (icns / ico / png)
│   └── renderer/
│       ├── index.html            # 메인 창
│       └── pages/                # 탭별 JS
│
├── data/
│   ├── raw/                      # 원본 Excel 데이터
│   └── processed/                # 전처리 결과
├── models/                       # 학습된 모델 .pkl (개발 환경)
├── outputs/                      # 시각화 / 예측 결과
├── notebooks/                    # 실험용 Jupyter 노트북
├── docs/
│   ├── 기술서.md
│   └── 계획서.md
├── requirements.txt
└── requirements-dev.txt
```

---

## 실행 방법

### 개발 환경

```bash
# 1. Python 의존성 설치
pip install -r requirements.txt

# 2. Flask 백엔드 실행
python python/main_api.py

# 3. Electron 의존성 설치 및 실행 (별도 터미널)
cd electron
npm install
npm start
```

### 배포 빌드

```bash
# Python 백엔드 단일 실행파일 생성
pyinstaller python/main_api.py --onefile --name main_api --distpath python/dist

# macOS DMG 생성
cd electron && npm run build:mac

# Windows EXE 생성
cd electron && npm run build:win
```

---

## API 엔드포인트

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/health` | GET | 서버 상태 확인 |
| `/preprocess/load` | POST | Excel 파일 로드 |
| `/preprocess/run` | POST | 전처리 실행 |
| `/preprocess/preview` | GET | 전처리 결과 미리보기 |
| `/preprocess/domain-ranges` | GET/PUT | 도메인 기준 조회/수정 |
| `/features/available` | GET | 선택 가능한 피처 컬럼 목록 |
| `/features/select` | POST | 학습에 사용할 피처 컬럼 지정 |
| `/features/selected` | GET | 현재 선택된 피처 컬럼 목록 |
| `/train/start` | POST | 모델 학습 시작 |
| `/train/status` | GET | 학습 진행 상태 (SSE) |
| `/performance/metrics` | GET | R², MAE 지표 |
| `/performance/parity` | GET | Parity Plot 데이터 |
| `/performance/heatmap` | GET | 상관관계 히트맵 데이터 |
| `/inference/predict` | POST | 4개 물성 예측 |

---

## 데이터 형식

- **입력**: `.xls` (Excel 97-2003) 및 `.xlsx` / `.xlsm` (Excel 2007+) — 원본 조성/공정 변수 29개 포함
- **모델 저장**: `joblib` 직렬화 `.pkl`
  - macOS: `~/Library/Application Support/AIMaterialsDiscovery/`
  - Windows: `%APPDATA%\AIMaterialsDiscovery\`

---

## 참고

- 도메인 기준은 [SSINA 오스테나이트 조성표](https://www.ssina.com/education/technical-resources/composition-properties/) 기반
- 세부 기술 명세: [`docs/기술서.md`](docs/기술서.md)
- 리팩토링 및 배포 계획: [`docs/계획서.md`](docs/계획서.md)
