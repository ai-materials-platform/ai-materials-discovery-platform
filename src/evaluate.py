import math
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def evaluate_regression(y_true, y_pred):
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": math.sqrt(mean_squared_error(y_true, y_pred)),
    }


def make_missing_report(df):
    report = pd.DataFrame(
        {
            "missing_count": df.isna().sum(),
            "missing_pct": (df.isna().mean() * 100).round(2),
            "dtype": df.dtypes.astype(str),
            "nunique": df.nunique(dropna=True),
        }
    ).sort_values("missing_pct", ascending=False)
    return report


def make_target_stats(df, cols):
    return df[cols].describe().T


def make_correlation_with_target(df, feature_cols, target_col, top_n=15):
    corr = df[feature_cols + [target_col]].corr(numeric_only=True)
    target_corr = (
        corr[target_col]
        .drop(target_col)
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .head(top_n)
    )
    return target_corr