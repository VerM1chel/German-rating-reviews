from pathlib import Path

VERSION = "patient_reviews"

CONFIGS = {
    "patient_reviews": {
        "csv_path": str(Path(__file__).parent.parent / "2021_german_doctor_reviews.csv"),
        "text_col": "comment",
        "label_col": "rating",
        "subset_size": 100000,
        "max_length": 128,
        "embedding_dim": 128,
        "hidden_dim": 128,
        "num_layers": 1,
        "dropout": 0.3,
        "batch_size": 64,
        "epochs": 5,
        "learning_rate": 1e-3,
        "seed": 42,
    },
}

CURRENT = CONFIGS[VERSION]