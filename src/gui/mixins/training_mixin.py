import os

import numpy as np

from src.gui.threads import TrainingThread


class TrainingMixin:
    def on_train_clicked(self):
        if not self.data_engine.file_path or not os.path.exists(self.data_engine.file_path):
            self.training_status_label.setText("상태: 오류 - 먼저 데이터 파일을 선택해 주세요.")
            return

        if not self.preprocessing_ready:
            self.training_status_label.setText("상태: 1번 탭에서 전처리를 먼저 실행해 주세요.")
            return

        selected_columns = self.data_engine.get_selected_training_columns(default_to_all=False)
        if not selected_columns:
            self.training_status_label.setText("상태: 오류 - 학습할 컬럼을 하나 이상 선택해 주세요.")
            self.tabs.setCurrentIndex(1)
            return

        self.apply_quality_settings_from_ui()
        self.train_btn.setEnabled(False)
        self.training_status_label.setText(
            f"상태: 학습 준비 중입니다. 선택한 컬럼 {len(selected_columns)}개만 사용합니다."
        )
        self.metrics_label.setText("<b>모델 성능 요약:</b><br>- 계산 중...")

        model_map = {0: "RF", 1: "GBM", 2: "MLP", 3: "TFP"}
        self.model_type = model_map.get(self.model_combo.currentIndex(), "RF")

        self.thread = TrainingThread(self.data_engine, model_type=self.model_type, max_iter=self.iter_spin.value())
        self.thread.progress.connect(lambda text: self.training_status_label.setText(f"상태: {text}"))
        self.thread.finished.connect(self.on_training_finished)
        self.thread.start()

    def on_training_finished(self, results):
        self.train_btn.setEnabled(True)
        if isinstance(results, str):
            self.training_status_label.setText(f"상태: 오류 발생 - {results}")
            return

        self.model_engine = results["model"]
        self.training_status_label.setText(f"상태: {self.model_type} 모델 학습이 완료되었습니다.")
        self.update_active_model_display()
        self.update_quality_summary_from_report(results.get("quality_report", {}))

        metrics = results["metrics"]
        r2_avg = float(np.mean(metrics["r2"]))
        mae_avg = float(np.mean(metrics["mae"]))
        self.last_r2_avg = round(r2_avg, 4)
        acc_text = "매우 높음" if r2_avg > 0.9 else "높음" if r2_avg > 0.8 else "보통"

        self.metrics_label.setText(
            f"<b>종합 모델 성능 요약:</b><br>- 평균 예측 정확도(R2): <b>{r2_avg * 100:.1f}% ({acc_text})</b><br>- 평균 오차(MAE): <b>{mae_avg:.2f}</b>"
        )

        self.canvas.axes.clear()
        target_names = ["Yield Stress", "UTS", "Elongation", "Area Red."]
        r2_scores = metrics["r2"]
        colors = ["#3498db" if score > 0.8 else "#f1c40f" if score > 0.6 else "#e74c3c" for score in r2_scores]
        bars = self.canvas.axes.bar(target_names, r2_scores, color=colors)
        self.canvas.axes.set_ylim(0, 1.1)
        self.canvas.axes.set_ylabel("정확도 (R2 Score)")
        for bar in bars:
            height = bar.get_height()
            self.canvas.axes.text(bar.get_x() + bar.get_width() / 2.0, height + 0.02, f"{height:.2f}", ha="center", va="bottom", fontsize=9)
        name_map = {"RF": "Random Forest", "GBM": "Gradient Boosting", "MLP": "Neural Network", "TFP": "TFP"}
        self.canvas.axes.set_title(f"모델별 특성 예측 정확도 ({name_map.get(self.model_type, self.model_type)})")
        self.canvas.draw()
        self.render_performance_results(results)

        self.auto_save_workspace()
        self._update_project_tree()
        self._sb_status.setText("● 학습 완료")
        self._sb_status.setStyleSheet("color: #58C472; font-size: 11px; font-weight: 700; padding: 0 10px;")

        self.append_log({
            "type": "학습",
            "model": self.model_type,
            "data_file": os.path.basename(self.data_engine.file_path or ""),
            "r2_avg": round(float(np.mean(metrics["r2"])), 4),
            "mae_avg": round(float(np.mean(metrics["mae"])), 4),
            "r2_per_target": [round(float(v), 4) for v in metrics["r2"]],
            "mae_per_target": [round(float(v), 4) for v in metrics["mae"]],
        })

    def render_performance_results(self, results):
        self.perf_canvas.figure.clear()
        axes = self.perf_canvas.figure.subplots(2, 2)
        y_test = results["y_test"].values
        y_pred = results["y_pred"]
        target_names = ["Yield Stress (MPa)", "UTS (MPa)", "Elongation (%)", "Area Reduction (%)"]
        colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]

        for index, ax in enumerate(axes.flatten()):
            ax.scatter(y_test[:, index], y_pred[:, index], alpha=0.55, color=colors[index], s=18)
            all_data = np.concatenate([y_test[:, index], y_pred[:, index]])
            min_val, max_val = all_data.min(), all_data.max()
            ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.7, lw=1)
            ax.set_title(target_names[index], fontsize=10, fontweight="bold")
            ax.set_xlabel("실제값", fontsize=9)
            ax.set_ylabel("예측값", fontsize=9)
            ax.grid(True, linestyle=":", alpha=0.6)

        self.perf_canvas.figure.tight_layout()
        self.perf_canvas.draw()
