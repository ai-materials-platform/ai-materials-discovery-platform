import json
import os

import joblib
import numpy as np

from src.engine.model_engine import ModelEngine


class InferenceMixin:
    def _run_prediction(
        self,
        model_engine,
        data_engine,
        inputs,
        result_label,
        canvas,
        curve_canvas=None,
        curve_label=None,
        prediction_state_attr=None,
        result_tabs=None,
        simulation_prefix=None,
    ):
        if not model_engine or not data_engine:
            result_label.setText(
                "<b>먼저 모델을 준비해 주세요.</b><br>"
                "학습을 완료하거나 저장된 모델을 불러온 뒤 예측을 실행할 수 있습니다."
            )
            return None

        input_dict = {key: widget.text() for key, widget in inputs.items()}
        scaled_input = data_engine.get_inference_data(input_dict)
        mean_scaled, std_scaled = model_engine.predict(scaled_input.astype(np.float32))

        mean = data_engine.scaler_y.inverse_transform(mean_scaled)[0]
        std = std_scaled[0] * data_engine.scaler_y.scale_

        note_color = self._theme()["text_label"]
        result_text = (
            f"<b>강도 예측 결과</b><br>"
            f"0.2% 항복강도: <b>{mean[0]:.1f} ± {std[0]:.1f} MPa</b><br>"
            f"인장강도(UTS): <b>{mean[1]:.1f} ± {std[1]:.1f} MPa</b><br><br>"
            f"<b>연성 예측 결과</b><br>"
            f"연신율: <b>{mean[2]:.1f} ± {std[2]:.1f} %</b><br>"
            f"단면감소율: <b>{mean[3]:.1f} ± {std[3]:.1f} %</b><br><br>"
            f"<span style='color:{note_color};'>Stress-Strain Curve 탭에서 예측 물성 기반 곡선을 확인할 수 있습니다.</span>"
        )
        result_label.setText(result_text)
        self._render_prediction_chart(canvas, mean, std)
        if curve_canvas is not None and curve_label is not None:
            self._render_stress_strain_curve(curve_canvas, curve_label, mean, input_dict)
        if simulation_prefix:
            self._render_simulation_view(simulation_prefix, mean, input_dict)
        if prediction_state_attr:
            setattr(
                self,
                prediction_state_attr,
                {
                    "mean": np.array(mean, dtype=float),
                    "std": np.array(std, dtype=float),
                    "input_dict": dict(input_dict),
                },
            )

        if result_tabs is not None:
            result_tabs.setCurrentIndex(0)

        return {
            "yield_stress": round(float(mean[0]), 2),
            "uts": round(float(mean[1]), 2),
            "elongation": round(float(mean[2]), 2),
            "area_reduction": round(float(mean[3]), 2),
        }

    def on_pretrained_predict_clicked(self):
        try:
            self._run_prediction(
                self.pretrained_model_engine,
                self.pretrained_data_engine,
                self.pretrained_inputs,
                self.pretrained_result_display,
                self.pretrained_prediction_canvas,
                self.pretrained_curve_canvas,
                self.pretrained_curve_placeholder,
                "_pretrained_prediction_state",
                self.pretrained_result_tabs,
                "pretrained",
            )
        except Exception as exc:
            self.pretrained_result_display.setText(f"<b>사전학습 모델 예측 중 오류가 발생했습니다.</b><br>{exc}")

    def on_predict_clicked(self):
        try:
            results = self._run_prediction(
                self.model_engine,
                self.data_engine,
                self.inputs,
                self.result_display,
                self.prediction_canvas,
                self.stress_strain_canvas,
                self.stress_strain_placeholder_label,
                "_user_prediction_state",
                self.inference_result_tabs,
                "user",
            )
            if results is None:
                return

            auto_folder = os.path.join("workspaces", "auto_save")
            if os.path.exists(auto_folder):
                self.prediction_canvas.fig.savefig(
                    os.path.join(auto_folder, "prediction.png"), dpi=200, bbox_inches="tight"
                )
                self.stress_strain_canvas.fig.savefig(
                    os.path.join(auto_folder, "stress_strain_curve.png"), dpi=200, bbox_inches="tight"
                )

            self.append_log({
                "type": "예측",
                "model": self.model_type,
                "inputs": {k: v.text() for k, v in self.inputs.items()},
                "results": results,
            })
        except Exception as exc:
            self.result_display.setText(f"<b>예측 중 오류가 발생했습니다.</b><br>{exc}")

    def prepare_pretrained_model(self):
        try:
            model_info = self._load_or_train_pretrained_model()
            self.pretrained_model_engine = model_info["model_engine"]
            self.pretrained_data_engine = model_info["data_engine"]
            self.pretrained_model_type = model_info["model_type"]
            self.pretrained_metrics = model_info["metrics"]

            name_map = {"RF": "Random Forest", "GBM": "Gradient Boosting", "MLP": "Neural Network", "TFP": "TFP"}
            model_name = name_map.get(self.pretrained_model_type, self.pretrained_model_type)
            self.pretrained_active_model_info.setText(
                f"사용 중인 모델: {model_name} | 평균 R2 {self.pretrained_metrics['r2_avg']:.3f} | 평균 MAE {self.pretrained_metrics['mae_avg']:.3f}"
            )
            self.pretrained_active_model_info.hide()
            self.pretrained_predict_btn.setEnabled(True)
        except Exception as exc:
            self.pretrained_predict_btn.setEnabled(False)
            self.pretrained_result_display.setText(
                "<b>사전학습 모델을 준비하지 못했습니다.</b><br>"
                f"{exc}"
            )
            self.pretrained_active_model_info.setText("사용 중인 모델: 준비되지 않음")
            self.pretrained_active_model_info.hide()

    def _load_or_train_pretrained_model(self):
        models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "models"))
        os.makedirs(models_dir, exist_ok=True)

        model_path = os.path.join(models_dir, "pretrained_material_model.pkl")
        data_engine_path = os.path.join(models_dir, "pretrained_data_engine.pkl")
        meta_path = os.path.join(models_dir, "pretrained_material_model_meta.json")
        required_files = {
            "model": model_path,
            "data_engine": data_engine_path,
            "meta": meta_path,
        }
        missing_files = [path for path in required_files.values() if not os.path.exists(path)]
        if missing_files:
            missing_names = ", ".join(os.path.basename(path) for path in missing_files)
            raise FileNotFoundError(
                "사전학습 예측 기능에 필요한 번들 모델 파일이 없습니다: "
                f"{missing_names}"
            )

        pretrained_engine = joblib.load(data_engine_path)
        pretrained_engine.file_path = None
        selected_columns = pretrained_engine.get_selected_training_columns()
        if "Fe" not in selected_columns:
            raise ValueError("사전학습 입력 컬럼 정보가 올바르지 않습니다.")

        pretrained_model = ModelEngine(model_type="RF", output_dim=4)
        pretrained_model.load(model_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        return {
            "model_engine": pretrained_model,
            "data_engine": pretrained_engine,
            "model_type": meta.get("model_type", pretrained_model.model_type),
            "metrics": meta,
        }

    def update_active_model_display(self):
        name_map = {"RF": "Random Forest", "GBM": "Gradient Boosting", "MLP": "Neural Network", "TFP": "TFP"}
        self.active_model_info.setText(f"현재 예측 모델: {name_map.get(self.model_type, self.model_type)}")
        if self.pretrained_model_type and hasattr(self, "pretrained_active_model_info"):
            metrics = self.pretrained_metrics or {}
            model_name = name_map.get(self.pretrained_model_type, self.pretrained_model_type)
            if metrics:
                self.pretrained_active_model_info.setText(
                    f"사용 중인 모델: {model_name} | 평균 R2 {metrics.get('r2_avg', 0):.3f} | 평균 MAE {metrics.get('mae_avg', 0):.3f}"
                )
            else:
                self.pretrained_active_model_info.setText(f"사용 중인 모델: {model_name}")
        self._apply_theme_colors()
