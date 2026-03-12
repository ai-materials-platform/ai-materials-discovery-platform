# 오스테나이트계 철강 고온강도 예측 구현 메모

## 반영한 요구사항 (PDF 기준)
- 데이터 전처리 파이프라인: 결측치/형변환 처리
- AI 학습 및 추론 기능: `train_models.py`, `predict_strength.py`
- 모델 저장/재사용: `models/*_best_pipeline.pkl`
- 성능 검증: Train/Test 분리 + 5-fold 교차검증 + MAE/RMSE/R²
- 중요도 분석: 모델별 feature importance 산출 및 시각화
- 신뢰도(불확실도) 산출: 앙상블 개별 트리 분산 기반 std/95% 구간 제공
- 결과 시각화 및 저장: `outputs/`에 그래프/CSV/요약 JSON 저장

## 현재 범위
- GUI(PyQt/Flask)는 이번 단계에서 제외하고, 데이터 분석과 예측 모델 구현에 집중함.
- 향후 GUI에서 `scripts/predict_strength.py`를 호출해 연결 가능.
