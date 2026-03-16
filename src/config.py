from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "STMECH_AUS_SS.xls"
README_PATH = BASE_DIR / "data" / "raw" / "readme.txt"

PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = OUTPUT_DIR / "reports"

TARGETS = [
    "0.2%proof_stress (M Pa)",
    "UTS (M Pa)",
    "Elongation (%)",
    "Area_reduction (%)",
]

PRIMARY_TARGETS = [
    "0.2%proof_stress (M Pa)",
    "UTS (M Pa)",
]

COMMENT_COLS = ["Comments"]

CATEGORICAL_COLS = [
    "Water_Quenched_after_s.t.",
    "Air_Quenched_after_s.t.",
    "Type of melting",
    "Product form",
]

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5