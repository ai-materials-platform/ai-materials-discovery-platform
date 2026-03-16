import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge

from .config import (
    PRIMARY_TARGETS,
    MODEL_DIR,
    REPORT_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    N_SPLITS,
)
from .preprocess import get_feature_columns, make_preprocessor
from .evaluate import evaluate_regression, make_correlation_with_target


def get_models():
    return {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            min_samples_leaf=2,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def train_one_target(df, target_name):
    feature_cols, numeric_cols, categorical_cols = get_feature_columns(df)

    X = df[feature_cols].copy()
    y = df[target_name].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    preprocessor = make_preprocessor(numeric_cols, categorical_cols)
    models = get_models()
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    results = []
    best_model_name = None
    best_pipe = None
    best_r2 = -999999

    for model_name, model in models.items():
        pipe = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", model),
            ]
        )

        cv_scores = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring=("r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"),
            n_jobs=-1,
        )

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        metrics = evaluate_regression(y_test, y_pred)

        results.append(
            {
                "target": target_name,
                "model": model_name,
                "cv_r2_mean": cv_scores["test_r2"].mean(),
                "cv_r2_std": cv_scores["test_r2"].std(),
                "cv_mae_mean": -cv_scores["test_neg_mean_absolute_error"].mean(),
                "cv_rmse_mean": -cv_scores["test_neg_root_mean_squared_error"].mean(),
                "test_r2": metrics["r2"],
                "test_mae": metrics["mae"],
                "test_rmse": metrics["rmse"],
            }
        )

        if metrics["r2"] > best_r2:
            best_r2 = metrics["r2"]
            best_model_name = model_name
            best_pipe = pipe

    results_df = pd.DataFrame(results).sort_values("test_r2", ascending=False)
    corr_df = make_correlation_with_target(df, feature_cols, target_name, top_n=15)

    return results_df, best_model_name, best_pipe, corr_df


def save_model(pipe, filename):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_DIR / filename)


def save_feature_importance(best_pipe, numeric_cols, categorical_cols, out_path):
    model = best_pipe.named_steps["model"]

    if not hasattr(model, "feature_importances_"):
        return None

    feature_names = []
    feature_names.extend(numeric_cols)

    if categorical_cols:
        ohe = (
            best_pipe.named_steps["preprocess"]
            .named_transformers_["cat"]
            .named_steps["onehot"]
        )
        feature_names.extend(list(ohe.get_feature_names_out(categorical_cols)))

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return importance_df


def train_all(df):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    for target in PRIMARY_TARGETS:
        results_df, best_model_name, best_pipe, corr_df = train_one_target(df, target)

        print(f"\n===== {target} =====")
        print(results_df)
        print(f"Best model: {best_model_name}")

        safe_name = "proof_stress" if "proof" in target else "uts"
        results_df.to_csv(REPORT_DIR / f"{safe_name}_model_results.csv", index=False, encoding="utf-8-sig")
        corr_df.to_csv(REPORT_DIR / f"{safe_name}_top_correlations.csv", encoding="utf-8-sig")

        feature_cols, numeric_cols, categorical_cols = get_feature_columns(df)
        save_feature_importance(
            best_pipe,
            numeric_cols,
            categorical_cols,
            REPORT_DIR / f"{safe_name}_feature_importance.csv",
        )

        save_model(best_pipe, f"{safe_name}_model.joblib")
        all_results.append(results_df)

    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv(REPORT_DIR / "all_model_results.csv", index=False, encoding="utf-8-sig")
    return final_df