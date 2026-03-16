from src.config import RAW_DATA_PATH, PROCESSED_DIR, REPORT_DIR, FIGURE_DIR
from src.data_loader import load_data, save_processed_data
from src.evaluate import make_missing_report, make_target_stats
from src.train import train_all


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(RAW_DATA_PATH)

    print("=== Data Shape ===")
    print(df.shape)

    processed_path = PROCESSED_DIR / "austenitic_clean.csv"
    save_processed_data(df, processed_path)

    missing_report = make_missing_report(df)
    target_stats = make_target_stats(
        df,
        [
            "0.2%proof_stress (M Pa)",
            "UTS (M Pa)",
            "Elongation (%)",
            "Area_reduction (%)",
            "Temperature (K)",
        ],
    )

    missing_report.to_csv(REPORT_DIR / "missing_report.csv", encoding="utf-8-sig")
    target_stats.to_csv(REPORT_DIR / "target_stats.csv", encoding="utf-8-sig")

    print("\n=== Missing Report Top 10 ===")
    print(missing_report.head(10))

    print("\n=== Target Statistics ===")
    print(target_stats)

    results = train_all(df)

    print("\n=== Final Results ===")
    print(results)


if __name__ == "__main__":
    main()