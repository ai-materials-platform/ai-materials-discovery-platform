from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

matplotlib.use("Agg")
sns.set_theme(style="whitegrid")


COLUMN_NAMES = [
    "chromium_wt_pct",
    "nickel_wt_pct",
    "molybdenum_wt_pct",
    "manganese_wt_pct",
    "silicon_wt_pct",
    "niobium_wt_pct",
    "titanium_wt_pct",
    "zirconium_wt_pct",
    "tantalum_wt_pct",
    "vanadium_wt_pct",
    "tungsten_wt_pct",
    "copper_wt_pct",
    "nitrogen_wt_pct",
    "carbon_wt_pct",
    "boron_wt_pct",
    "phosphorus_wt_pct",
    "sulphur_wt_pct",
    "cobalt_wt_pct",
    "aluminium_wt_pct",
    "tin_wt_pct",
    "lead_wt_pct",
    "solution_treatment_temp_k",
    "solution_treatment_time_s",
    "water_quenched",
    "air_quenched",
    "grains_per_mm2",
    "type_of_melting",
    "size_of_ingot",
    "product_form",
    "test_temp_k",
    "proof_stress_mpa",
    "uts_mpa",
    "elongation_pct",
    "area_reduction_pct",
    "comments",
]

TARGETS = ["proof_stress_mpa", "uts_mpa"]
LEAKAGE_COLUMNS = ["elongation_pct", "area_reduction_pct", "comments"]
CATEGORICAL_COLUMNS = ["type_of_melting", "size_of_ingot", "product_form"]


@dataclass
class TrainingArtifacts:
    target: str
    best_model_name: str
    best_pipeline: Pipeline
    test_split: dict[str, pd.DataFrame | pd.Series]
    metrics_df: pd.DataFrame
    feature_importance_df: pd.DataFrame
    predictions_df: pd.DataFrame


def load_and_clean_data(raw_path: str | Path) -> pd.DataFrame:
    """Load the xls dataset and normalize data types."""
    df = pd.read_excel(raw_path, header=5, names=COLUMN_NAMES)

    for col in COLUMN_NAMES:
        if col == "comments":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # The targets are fully observed in this dataset. Keep rows only if present.
    df = df.dropna(subset=TARGETS).reset_index(drop=True)
    return df


def make_feature_matrix(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    drop_cols = [target] + [t for t in TARGETS if t != target] + LEAKAGE_COLUMNS
    X = df.drop(columns=drop_cols)
    y = df[target].copy()
    return X, y


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = [c for c in X.columns if c not in CATEGORICAL_COLUMNS]
    categorical_cols = [c for c in CATEGORICAL_COLUMNS if c in X.columns]

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ]
    )


def build_model_candidates(random_state: int = 42) -> dict[str, Any]:
    return {
        "random_forest": RandomForestRegressor(
            n_estimators=220,
            max_depth=None,
            min_samples_split=2,
            random_state=random_state,
            n_jobs=1,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=220,
            max_depth=None,
            min_samples_split=2,
            random_state=random_state,
            n_jobs=1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=220,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
        ),
    }


def _rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _extract_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        if name == "num":
            names.extend(cols)
            continue
        if name == "cat":
            onehot = trans.named_steps["onehot"]
            cat_names = onehot.get_feature_names_out(cols)
            names.extend(cat_names.tolist())
    return names


def _feature_importance(best_pipeline: Pipeline) -> pd.DataFrame:
    pre = best_pipeline.named_steps["preprocessor"]
    reg = best_pipeline.named_steps["regressor"]
    feature_names = _extract_feature_names(pre)

    importances = None
    if hasattr(reg, "feature_importances_"):
        importances = reg.feature_importances_
    else:
        importances = np.zeros(len(feature_names), dtype=float)

    fi = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)
    return fi


def train_single_target(
    df: pd.DataFrame,
    target: str,
    random_state: int = 42,
    test_size: float = 0.2,
) -> TrainingArtifacts:
    X, y = make_feature_matrix(df, target=target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    preprocessor = make_preprocessor(X_train)
    model_candidates = build_model_candidates(random_state=random_state)
    cv = KFold(n_splits=3, shuffle=True, random_state=random_state)

    rows = []
    best_name = None
    best_pipeline = None
    best_rmse = float("inf")

    for model_name, regressor in model_candidates.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", regressor),
            ]
        )
        cv_rmse = -cross_val_score(
            pipeline,
            X_train,
            y_train,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=1,
        )
        cv_r2 = cross_val_score(
            pipeline,
            X_train,
            y_train,
            scoring="r2",
            cv=cv,
            n_jobs=1,
        )
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        rmse = _rmse(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)

        rows.append(
            {
                "target": target,
                "model": model_name,
                "cv_rmse_mean": float(np.mean(cv_rmse)),
                "cv_rmse_std": float(np.std(cv_rmse)),
                "cv_r2_mean": float(np.mean(cv_r2)),
                "cv_r2_std": float(np.std(cv_r2)),
                "test_rmse": float(rmse),
                "test_mae": float(mae),
                "test_r2": float(r2),
            }
        )

        if rmse < best_rmse:
            best_rmse = rmse
            best_name = model_name
            best_pipeline = pipeline

    metrics_df = pd.DataFrame(rows).sort_values("test_rmse").reset_index(drop=True)
    assert best_name is not None and best_pipeline is not None

    best_pred = best_pipeline.predict(X_test)
    pred_df = pd.DataFrame(
        {
            "target": target,
            "actual": y_test.to_numpy(),
            "predicted": best_pred,
            "abs_error": np.abs(y_test.to_numpy() - best_pred),
        }
    )
    fi_df = _feature_importance(best_pipeline)

    return TrainingArtifacts(
        target=target,
        best_model_name=best_name,
        best_pipeline=best_pipeline,
        test_split={
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        },
        metrics_df=metrics_df,
        feature_importance_df=fi_df,
        predictions_df=pred_df,
    )


def _ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_artifacts(
    artifacts: TrainingArtifacts,
    model_dir: str | Path,
    output_dir: str | Path,
) -> None:
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{artifacts.target}_best_pipeline.pkl"
    joblib.dump(artifacts.best_pipeline, model_path)

    metrics_path = output_dir / f"{artifacts.target}_metrics.csv"
    pred_path = output_dir / f"{artifacts.target}_test_predictions.csv"
    fi_path = output_dir / f"{artifacts.target}_feature_importance.csv"
    artifacts.metrics_df.to_csv(metrics_path, index=False)
    artifacts.predictions_df.to_csv(pred_path, index=False)
    artifacts.feature_importance_df.to_csv(fi_path, index=False)


def plot_eda(df: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Target distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df["proof_stress_mpa"], kde=True, ax=axes[0], color="#2c7fb8")
    axes[0].set_title("0.2% Proof Stress Distribution")
    axes[0].set_xlabel("MPa")
    sns.histplot(df["uts_mpa"], kde=True, ax=axes[1], color="#f03b20")
    axes[1].set_title("UTS Distribution")
    axes[1].set_xlabel("MPa")
    fig.tight_layout()
    fig.savefig(output_dir / "1_target_distributions.png", dpi=200)
    plt.close(fig)

    # 2) Temperature vs target trends
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.scatterplot(data=df, x="test_temp_k", y="proof_stress_mpa", alpha=0.55, ax=axes[0])
    axes[0].set_title("Temperature vs Proof Stress")
    sns.scatterplot(data=df, x="test_temp_k", y="uts_mpa", alpha=0.55, ax=axes[1], color="#ef6548")
    axes[1].set_title("Temperature vs UTS")
    fig.tight_layout()
    fig.savefig(output_dir / "2_temperature_vs_strength.png", dpi=200)
    plt.close(fig)

    # 3) Correlation heatmap for key numeric columns
    key_cols = [
        "chromium_wt_pct",
        "nickel_wt_pct",
        "molybdenum_wt_pct",
        "nitrogen_wt_pct",
        "carbon_wt_pct",
        "solution_treatment_temp_k",
        "test_temp_k",
        "proof_stress_mpa",
        "uts_mpa",
    ]
    corr = df[key_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8.5, 6))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap (Key Variables)")
    fig.tight_layout()
    fig.savefig(output_dir / "3_correlation_heatmap.png", dpi=200)
    plt.close(fig)


def plot_model_comparison(all_metrics: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=all_metrics, x="model", y="test_rmse", hue="target", ax=ax)
    ax.set_title("Model Comparison by Target (Lower RMSE is Better)")
    ax.set_ylabel("Test RMSE")
    fig.tight_layout()
    fig.savefig(output_dir / "4_model_performance.png", dpi=200)
    plt.close(fig)


def plot_feature_importance(
    feature_importance_df: pd.DataFrame,
    target: str,
    output_dir: str | Path,
    top_k: int = 15,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    top = feature_importance_df.head(top_k).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"], top["importance"], color="#3182bd")
    ax.set_title(f"Top {top_k} Feature Importances ({target})")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(output_dir / f"5_feature_importance_{target}.png", dpi=200)
    plt.close(fig)


def predict_with_uncertainty(
    pipeline: Pipeline,
    X: pd.DataFrame,
    confidence: float = 0.95,
) -> pd.DataFrame:
    mean_pred = pipeline.predict(X)
    reg = pipeline.named_steps["regressor"]

    if hasattr(reg, "estimators_"):
        transformed = pipeline.named_steps["preprocessor"].transform(X)
        estimators = reg.estimators_
        if isinstance(estimators, np.ndarray):
            estimators = estimators.ravel().tolist()
        preds = []
        for est in estimators:
            if hasattr(est, "predict"):
                preds.append(est.predict(transformed))
        if preds:
            member_preds = np.column_stack(preds)
            std = member_preds.std(axis=1)
        else:
            std = np.full_like(mean_pred, fill_value=np.nan, dtype=float)
    else:
        std = np.full_like(mean_pred, fill_value=np.nan, dtype=float)

    z = 1.96 if confidence == 0.95 else 1.0
    lower = mean_pred - z * std
    upper = mean_pred + z * std

    return pd.DataFrame(
        {
            "prediction": mean_pred,
            "uncertainty_std": std,
            "lower_95": lower,
            "upper_95": upper,
        }
    )
