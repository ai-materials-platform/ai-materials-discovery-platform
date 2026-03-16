from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import TARGETS, COMMENT_COLS, CATEGORICAL_COLS


def get_feature_columns(df):
    feature_cols = [c for c in df.columns if c not in COMMENT_COLS + TARGETS]
    categorical_cols = [c for c in CATEGORICAL_COLS if c in feature_cols]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]
    return feature_cols, numeric_cols, categorical_cols


def make_preprocessor(numeric_cols, categorical_cols):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    return preprocessor