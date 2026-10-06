from pathlib import Path

VERSION = "patient_reviews"

DATASETS = {
    "patient_reviews": {
        "source": "local",
        "path": str(Path(__file__).parent.parent / "2021_german_doctor_reviews.csv"),
        "test_path": None,
        "lang": "de",
        "text_col": "comment",
        "label_col": "rating",
        "task": "classification",
        "max_features": 20000,
        "stratify": True,  # False для задач без меток (Оставить ли для будущих проектов?)
    },
}

CURRENT = DATASETS[VERSION]

