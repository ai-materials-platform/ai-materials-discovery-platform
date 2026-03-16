# 데이터 설명

## 데이터 파일
- STMECH_AUS_SS.xlsx
- readme.txt

## 주요 입력 변수
- Cr, Ni, Mo, Mn, Si, Nb, Ti 등 화학 조성
- Solution treatment temperature
- Solution treatment time
- Water quenched / Air quenched
- Grains mm-2
- Type of melting
- Size of ingot
- Product form
- Temperature (K)

## 주요 출력 변수
- 0.2%proof_stress (M Pa)
- UTS (M Pa)
- Elongation (%)
- Area_reduction (%)

## 전처리 방법
- "Na", "NA", 공백 문자열을 결측치로 처리
- 수치형 변수는 중앙값으로 대체
- 범주형 변수는 최빈값으로 대체 후 원-핫 인코딩