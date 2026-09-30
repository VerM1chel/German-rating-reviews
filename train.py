from config import CURRENT
from data_loader import load_data, vectorize
from clean import clean
from models import get_logreg, get_xgb

from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight


# Загрузка
train, test = load_data()

# Очистка
train["Text"] = train[CURRENT["text_col"]].apply(clean)
test["Text"] = test[CURRENT["text_col"]].apply(clean)
train = train.dropna(subset=[CURRENT["label_col"]])
test = test.dropna(subset=[CURRENT["label_col"]])

# Векторизация
X_train, X_test, vect = vectorize(train, test)

# Метки
y_train = train[CURRENT["label_col"]]
y_test_true = test[CURRENT["label_col"]]


# LogisticRegression
model_lr = get_logreg()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

print(f"Accuracy LR: {accuracy_score(y_test_true, y_pred_lr):.4f}")
print(f"F1 macro LR: {f1_score(y_test_true, y_pred_lr, average='macro'):.4f}")


# XGBoost
y_train_xgb = y_train - 1
y_test_xgb = y_test_true - 1

model_xgb = get_xgb()
weights = compute_sample_weight(class_weight='balanced', y=y_train_xgb)
model_xgb.fit(
    X_train, y_train_xgb,
    sample_weight=weights,
    eval_set=[(X_train, y_train_xgb), (X_test, y_test_xgb)],
    verbose=10
)

y_pred_xgb = model_xgb.predict(X_test)

print(f"Accuracy XGB: {accuracy_score(y_test_xgb, y_pred_xgb):.4f}")
print(f"F1 macro XGB: {f1_score(y_test_xgb, y_pred_xgb, average='macro'):.4f}")