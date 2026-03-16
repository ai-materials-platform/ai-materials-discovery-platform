import joblib
from .config import MODEL_DIR


def load_model(model_name):
    return joblib.load(MODEL_DIR / model_name)


def predict_strength(input_df):
    proof_model = load_model("proof_stress_model.joblib")
    uts_model = load_model("uts_model.joblib")

    proof_pred = proof_model.predict(input_df)[0]
    uts_pred = uts_model.predict(input_df)[0]

    return {
        "predicted_0.2%proof_stress (M Pa)": float(proof_pred),
        "predicted_UTS (M Pa)": float(uts_pred),
    }