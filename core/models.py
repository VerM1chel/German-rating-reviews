from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
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

def get_mlp():
    return MLPClassifier(
        hidden_layer_sizes=(100,),
        max_iter=200,
        random_state=42
    )

def get_svm():
    return SVC(
        kernel='rbf',
        class_weight='balanced',
        random_state=42,
    )