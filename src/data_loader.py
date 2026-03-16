import pandas as pd
import numpy as np


def load_data(path):
    """
    Excel 데이터를 읽고 기본 정리를 수행한다.
    """
    df = pd.read_excel(path, sheet_name="alldata", header=5)

    # 문자열 결측 처리
    df = df.replace(["Na", "NA", "na", ""], np.nan)

    # object 타입 중 숫자로 변환 가능한 컬럼은 숫자형으로 변환
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                pass

    return df


def save_processed_data(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")