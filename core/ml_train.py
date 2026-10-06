import time
from core.config import CURRENT
from core.data_loader import load_data, vectorize
from core.clean import clean
from core.models import get_logreg

from sklearn.metrics import accuracy_score, f1_score, classification_report

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
start = time.time()
model_lr.fit(X_train, y_train)
lr_train = time.time() - start

start = time.time()
y_pred_lr = model_lr.predict(X_test)
lr_inf = time.time() - start
print(f"LR — Train: {lr_train:.2f}s, Inference: {lr_inf:.4f}s")
print(f"LR — Accuracy: {accuracy_score(y_test_true, y_pred_lr):.4f}")
print(f"LR — F1 macro: {f1_score(y_test_true, y_pred_lr, average='macro'):.4f}")

print("\nLR — Per-class report:")
print(classification_report(y_test_true, y_pred_lr, digits=4))

# # XGBoost
# y_train_xgb = y_train - 1
# y_test_xgb = y_test_true - 1
#
# model_xgb = get_xgb()
# weights = compute_sample_weight(class_weight='balanced', y=y_train_xgb)
#
# start = time.time()
# model_xgb.fit(
#     X_train, y_train_xgb,
#     sample_weight=weights,
#     eval_set=[(X_train, y_train_xgb), (X_test, y_test_xgb)],
#     verbose=10
# )
# xgb_train = time.time() - start
#
# start = time.time()
# y_pred_xgb = model_xgb.predict(X_test)
# xgb_inf = time.time() - start
#
# print(f"XGB — Train: {xgb_train:.2f}s, Inference: {xgb_inf:.4f}s")
# print(f"XGB — Accuracy: {accuracy_score(y_test_xgb, y_pred_xgb):.4f}")
# print(f"XGB — F1 macro: {f1_score(y_test_xgb, y_pred_xgb, average='macro'):.4f}")
#
#
# # MLP
# model_mlp = get_mlp()
# start = time.time()
# model_mlp.fit(X_train, y_train)
# mlp_train = time.time() - start
#
# start = time.time()
# y_pred_mlp = model_mlp.predict(X_test)
# mlp_inf = time.time() - start
#
# print(f"MLP — Train: {mlp_train:.2f}s, Inference: {mlp_inf:.4f}s")
# print(f"MLP — Accuracy: {accuracy_score(y_test_true, y_pred_mlp):.4f}")
# print(f"MLP — F1 macro: {f1_score(y_test_true, y_pred_mlp, average='macro'):.4f}")