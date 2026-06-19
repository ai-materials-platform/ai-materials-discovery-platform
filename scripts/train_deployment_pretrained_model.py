import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.engine.data_engine import DataEngine
from src.engine.model_engine import ModelEngine


def _default_data_path():
    candidates = [
        REPO_ROOT / "data" / "raw" / "STMECH_AUS_SS.xls",
        REPO_ROOT / "data" / "raw" / "STMECH_AUS_SS.xlsx",
        Path.home() / "Downloads" / "STMECH_AUS_SS.xls",
        Path.home() / "Downloads" / "STMECH_AUS_SS.xlsx",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("STMECH_AUS_SS.xls[x] 파일을 data/raw 또는 Downloads에서 찾을 수 없습니다.")


def train(data_path, model_type, output_dir):
    data_engine = DataEngine(str(data_path))
    data_engine.configure_deployment_profile()
    data_engine.load_data()

    X_train, X_test, y_train, _, _, y_raw_test = data_engine.preprocess_data()
    model_engine = ModelEngine(model_type=model_type, output_dim=y_train.shape[1])
    model_engine.train(X_train, y_train)

    mean_scaled, _ = model_engine.predict(X_test)
    y_pred = data_engine.inverse_transform_y(mean_scaled)
    r2 = r2_score(y_raw_test, y_pred, multioutput="raw_values")
    mae = mean_absolute_error(y_raw_test, y_pred, multioutput="raw_values")

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "pretrained_material_model.pkl"
    data_engine_path = output_dir / "pretrained_data_engine.pkl"
    meta_path = output_dir / "pretrained_material_model_meta.json"

    model_engine.save(model_path)
    joblib.dump(data_engine, data_engine_path)

    selected_features = data_engine.get_selected_training_columns()
    target_cols = data_engine.get_active_target_columns(data_engine.df)
    meta = {
        "model_type": model_type,
        "training_profile": "deployment",
        "data_file": os.path.basename(str(data_path)),
        "input_policy": "composition_plus_test_temperature",
        "feature_columns": selected_features,
        "target_columns": target_cols,
        "target_display_names": data_engine.get_active_target_display_names(data_engine.df),
        "r2_avg": round(float(np.mean(r2)), 4),
        "mae_avg": round(float(np.mean(mae)), 4),
        "r2_per_target": [round(float(v), 4) for v in r2],
        "mae_per_target": [round(float(v), 4) for v in mae],
        "last_updated": date.today().isoformat(),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta


def main():
    parser = argparse.ArgumentParser(description="Train the bundled deployment model.")
    parser.add_argument("--data", type=Path, default=None, help="Path to STMECH_AUS_SS.xls[x].")
    parser.add_argument("--model-type", default="RF", choices=["RF", "GBM", "MLP", "TFP"])
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    args = parser.parse_args()

    data_path = args.data or _default_data_path()
    meta = train(data_path, args.model_type, args.output_dir)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
