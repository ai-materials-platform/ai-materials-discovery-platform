from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

from materials_ai.pipeline import predict_with_uncertainty


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for trained strength model.")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained pipeline pkl file.",
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        required=True,
        help="CSV with feature columns used during training.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="outputs/predictions.csv",
        help="Path to save predictions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = joblib.load(args.model_path)
    X = pd.read_csv(args.input_csv)
    pred_df = predict_with_uncertainty(pipeline, X)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_path, index=False)
    print(f"Saved predictions: {out_path}")
    print(pred_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
