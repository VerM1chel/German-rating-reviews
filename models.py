from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def get_logreg():
    return LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced"
    )


def get_xgb():
    return XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss'
    )