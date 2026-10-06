from pathlib import Path

VERSION = "patient_reviews"

CONFIGS = {
    "patient_reviews": {
        "csv_path": str(Path(__file__).parent.parent / "2021_german_doctor_reviews.csv"),
        "text_col": "comment",
        "label_col": "rating",
        "model_name": "distilbert-base-german-cased",
        "subset_size": 80000,
        "max_length": 128,
        "batch_size": 16,
        "epochs": 2,
        "seed": 42,
        "output_dir": "./distilbert_german_reviews",
    },
}

CURRENT = CONFIGS[VERSION]