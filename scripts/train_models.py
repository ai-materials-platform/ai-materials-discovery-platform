from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

from materials_ai.pipeline import (
    TARGETS,
    load_and_clean_data,
    plot_eda,
    plot_feature_importance,
    plot_model_comparison,
    save_artifacts,
    train_single_target,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train high-temperature strength prediction models for austenitic steel."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw/STMECH_AUS_SS.xls",
        help="Path to the raw xls dataset.",
    )
    parser.add_argument(
        "--processed-path",
        type=str,
        default="data/processed/aus_ss_clean.csv",
        help="Path to save cleaned dataset.",
    )
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = load_and_clean_data(args.data_path)
    processed_path = Path(args.processed_path)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)

    plot_eda(df, args.output_dir)

    all_metrics = []
    summary = {"rows": int(df.shape[0]), "columns": int(df.shape[1]), "targets": {}}

    for target in TARGETS:
        artifacts = train_single_target(df, target=target, random_state=args.random_state)
        save_artifacts(artifacts, model_dir=args.model_dir, output_dir=args.output_dir)
        plot_feature_importance(artifacts.feature_importance_df, target, args.output_dir)
        all_metrics.append(artifacts.metrics_df)

        summary["targets"][target] = {
            "best_model": artifacts.best_model_name,
            "best_test_rmse": float(artifacts.metrics_df.iloc[0]["test_rmse"]),
            "best_test_mae": float(artifacts.metrics_df.iloc[0]["test_mae"]),
            "best_test_r2": float(artifacts.metrics_df.iloc[0]["test_r2"]),
        }

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    metrics_df.to_csv(Path(args.output_dir) / "model_metrics_all.csv", index=False)
    plot_model_comparison(metrics_df, args.output_dir)

    with open(Path(args.output_dir) / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Training complete.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
