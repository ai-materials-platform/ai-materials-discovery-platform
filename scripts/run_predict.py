from src.config import RAW_DATA_PATH
from src.data_loader import load_data
from src.preprocess import get_feature_columns
from src.predict import predict_strength


def main():
    df = load_data(RAW_DATA_PATH)
    feature_cols, _, _ = get_feature_columns(df)

    sample_input = df[feature_cols].iloc[[0]].copy()
    result = predict_strength(sample_input)

    print("=== Sample Prediction ===")
    print(result)


if __name__ == "__main__":
    main()