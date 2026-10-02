VERSION = "patient_reviews"

DATASETS = {
    "patient_reviews": {
        "source": "local",
        "path": "2021_german_doctor_reviews.csv",
        "test_path": None,
        "lang": "de",
        "text_col": "comment",
        "label_col": "rating",
        "task": "classification",
        "max_features": 20000,
    },
}

CURRENT = DATASETS[VERSION]

