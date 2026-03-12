from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, KFold, cross_val_score
from sklearn.pipeline import Pipeline

from materials_ai.pipeline import (
    CATEGORICAL_COLUMNS,
    TARGETS,
    load_and_clean_data,
    make_feature_matrix,
    make_preprocessor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Advanced training with Optuna + high-temp evaluation.")
    parser.add_argument("--data-path", default="data/raw/STMECH_AUS_SS.xls")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--high-temp-threshold", type=float, default=800.0)
    return parser.parse_args()


def _rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["cr_eq"] = (
        out["chromium_wt_pct"]
        + out["molybdenum_wt_pct"]
        + 1.5 * out["silicon_wt_pct"]
        + 0.5 * out["niobium_wt_pct"]
    )
    out["ni_eq"] = (
        out["nickel_wt_pct"]
        + 30.0 * out["carbon_wt_pct"]
        + 30.0 * out["nitrogen_wt_pct"]
        + 0.5 * out["manganese_wt_pct"]
    )
    out["cr_ni_ratio"] = out["chromium_wt_pct"] / (out["nickel_wt_pct"] + 1e-6)
    out["temp_sq"] = out["test_temp_k"] ** 2
    out["mo_w_sum"] = out["molybdenum_wt_pct"] + out["tungsten_wt_pct"]
    out["c_n_sum"] = out["carbon_wt_pct"] + out["nitrogen_wt_pct"]
    return out


def build_group_id(X: pd.DataFrame) -> pd.Series:
    core_cols = [
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
        "type_of_melting",
        "size_of_ingot",
        "product_form",
    ]
    core_cols = [c for c in core_cols if c in X.columns]
    sig = X[core_cols].copy()
    for c in sig.columns:
        sig[c] = pd.to_numeric(sig[c], errors="coerce").round(4)
    return sig.astype(str).agg("|".join, axis=1)


def make_model(model_name: str, params: dict[str, Any], random_state: int) -> Any:
    if model_name == "random_forest":
        return RandomForestRegressor(
            random_state=random_state,
            n_jobs=1,
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
        )
    if model_name == "extra_trees":
        return ExtraTreesRegressor(
            random_state=random_state,
            n_jobs=1,
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
        )
    if model_name == "gradient_boosting":
        return GradientBoostingRegressor(
            random_state=random_state,
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            subsample=params["subsample"],
        )
    raise ValueError(f"Unsupported model: {model_name}")


def suggest_params(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    if model_name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 500),
            "max_depth": trial.suggest_int("max_depth", 4, 24),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 12),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
        }
    if model_name == "extra_trees":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 500),
            "max_depth": trial.suggest_int("max_depth", 4, 24),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 12),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
        }
    if model_name == "gradient_boosting":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 120, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 12),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        }
    raise ValueError(f"Unsupported model: {model_name}")


def evaluate_target(
    df: pd.DataFrame,
    target: str,
    output_dir: Path,
    model_dir: Path,
    random_state: int,
    trials: int,
    high_temp_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    X, y = make_feature_matrix(df, target)
    X = add_engineered_features(X)

    for cat_col in CATEGORICAL_COLUMNS:
        if cat_col in X.columns:
            X[cat_col] = X[cat_col].fillna(-1).astype(str)

    groups = build_group_id(X)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()

    preprocessor = make_preprocessor(X_train)
    cv = KFold(n_splits=3, shuffle=True, random_state=random_state)

    rows = []
    best = {"model": None, "rmse": float("inf"), "pipeline": None, "params": None}
    best_params_by_model: dict[str, Any] = {}

    for model_name in ["gradient_boosting", "extra_trees", "random_forest"]:
        study = optuna.create_study(direction="minimize")

        def objective(trial: optuna.Trial) -> float:
            params = suggest_params(trial, model_name)
            reg = make_model(model_name, params, random_state=random_state)
            pipe = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("regressor", reg),
                ]
            )
            scores = -cross_val_score(
                pipe,
                X_train,
                y_train,
                cv=cv,
                scoring="neg_root_mean_squared_error",
                n_jobs=1,
            )
            return float(np.mean(scores))

        study.optimize(objective, n_trials=trials, show_progress_bar=False)
        params = study.best_params
        best_params_by_model[model_name] = params
        reg = make_model(model_name, params, random_state=random_state)
        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", reg),
            ]
        )
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rmse = _rmse(y_test, pred)
        mae = float(mean_absolute_error(y_test, pred))
        r2 = float(r2_score(y_test, pred))
        rows.append(
            {
                "target": target,
                "model": model_name,
                "cv_best_rmse": float(study.best_value),
                "test_rmse": rmse,
                "test_mae": mae,
                "test_r2": r2,
            }
        )
        if rmse < best["rmse"]:
            best = {"model": model_name, "rmse": rmse, "pipeline": pipe, "params": params}

    metrics_df = pd.DataFrame(rows).sort_values("test_rmse").reset_index(drop=True)

    # High-temperature evaluation on test split only
    high_mask = X_test["test_temp_k"] >= high_temp_threshold
    high_rows = []
    if high_mask.sum() > 10:
        for _, row in metrics_df.iterrows():
            model_name = row["model"]
            reg = make_model(model_name, best_params_by_model[model_name], random_state=random_state)
            pipe = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("regressor", reg),
                ]
            )
            pipe.fit(X_train, y_train)
            hp = pipe.predict(X_test.loc[high_mask])
            hy = y_test.loc[high_mask]
            high_rows.append(
                {
                    "target": target,
                    "model": model_name,
                    "threshold_k": high_temp_threshold,
                    "n_samples": int(high_mask.sum()),
                    "high_temp_rmse": _rmse(hy, hp),
                    "high_temp_mae": float(mean_absolute_error(hy, hp)),
                    "high_temp_r2": float(r2_score(hy, hp)),
                }
            )
    high_df = pd.DataFrame(high_rows)

    model_path = model_dir / f"{target}_advanced_best_pipeline.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(best["pipeline"], model_path)

    pred_df = pd.DataFrame(
        {
            "actual": y_test.to_numpy(),
            "predicted": best["pipeline"].predict(X_test),
            "test_temp_k": X_test["test_temp_k"].to_numpy(),
        }
    )
    pred_df["abs_error"] = np.abs(pred_df["actual"] - pred_df["predicted"])
    pred_df.to_csv(output_dir / f"{target}_advanced_predictions.csv", index=False)

    return metrics_df, high_df, {"target": target, "best": best, "best_params_by_model": best_params_by_model}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_clean_data(args.data_path)

    metrics_all = []
    high_all = []
    summary: dict[str, Any] = {"targets": {}}

    for target in TARGETS:
        metrics_df, high_df, info = evaluate_target(
            df=df,
            target=target,
            output_dir=output_dir,
            model_dir=model_dir,
            random_state=args.random_state,
            trials=args.trials,
            high_temp_threshold=args.high_temp_threshold,
        )
        metrics_all.append(metrics_df)
        if not high_df.empty:
            high_all.append(high_df)

        best_row = metrics_df.iloc[0].to_dict()
        summary["targets"][target] = {
            "best_model": best_row["model"],
            "test_rmse": best_row["test_rmse"],
            "test_mae": best_row["test_mae"],
            "test_r2": best_row["test_r2"],
            "best_params": info["best_params_by_model"][best_row["model"]],
        }

    pd.concat(metrics_all, ignore_index=True).to_csv(output_dir / "advanced_metrics_all.csv", index=False)
    if high_all:
        pd.concat(high_all, ignore_index=True).to_csv(output_dir / "advanced_high_temp_metrics.csv", index=False)

    with open(output_dir / "advanced_training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
