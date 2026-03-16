# 모델 설계

## 예측 목표
오스테나이트계 철강의 고온 강도 예측

## 선택한 타깃
- 0.2%proof_stress (M Pa)
- UTS (M Pa)

## 사용 모델
1. Ridge Regression
2. Random Forest Regressor
3. Extra Trees Regressor

## 평가 방법
- Train/Test Split: 8:2
- K-Fold Cross Validation: 5-fold
- 평가지표:
  - R²
  - MAE
  - RMSE

## 설계 이유
- 선형성과 비선형성을 모두 비교하기 위해 Ridge와 트리 앙상블 모델을 함께 사용
- 트리 기반 모델은 조성, 공정, 시험온도의 복합 비선형 관계를 잘 반영할 수 있음

## 확장 방향
- Flask 기반 추론 API 연결
- PyQt6 GUI 연결
- 불확실성 추정을 위한 추가 모델 적용