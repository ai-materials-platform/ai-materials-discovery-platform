# AI Materials Discovery Platform - 사용자 가이드

본 플랫폼은 전산재료 데이터와 인공지능 기술을 융합하여 신소재의 기계적 물성을 예측하고 분석하기 위해 개발되었습니다.

## 주요 기능
1. **데이터 자동 추출 및 정량화**: `data/STMECH_AUS_SS.xls` 파일을 자동으로 읽어와 학습 데이터로 변환합니다.
2. **TFP 기반 확률론적 모델링**: 단순 예측뿐만 아니라 예측값의 **신뢰도(불확실성)**를 함께 산출합니다.
3. **학습용 GUI**: 데이터 학습 상황, R2, MAE 지표 및 예측 성능 그래프를 실시간으로 확인합니다.
4. **추론용 GUI**: 합금 조성 및 공정 조건을 입력하여 즉시 UTS와 항복강도를 예측합니다.

## 실행 방법
```bash
pip install -r requirements.txt  # (필요한 라이브러리들)
python main.py
```

## 시스템 구성
- **Data Engine**: 엑셀 데이터의 전처리, 정규화 및 인코딩 담당
- **Model Engine**: TensorFlow Probability 기반의 심층 신경망 모델
- **Inference System**: Flask API 서버 및 PyQt6 GUI를 통한 결과 도출

## 참고 사항
- 모델 학습 시 `models/` 폴더에 학습된 가중치와 데이터 스케일러가 저장됩니다.
- 신소재 탐색 시, 'Property Inference' 탭에서 다양한 조건(온도, 공정 등)을 입력하여 물성 변화를 시뮬레이션할 수 있습니다.

