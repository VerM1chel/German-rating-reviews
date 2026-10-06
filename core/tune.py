import time

from core.config import CURRENT
from core.data_loader import load_data
from core.clean import clean
from search_config import search_mlp

# Загрузка
train, test = load_data()

# Очистка
train["Text"] = train[CURRENT["text_col"]].apply(clean)
test["Text"] = test[CURRENT["text_col"]].apply(clean)
train = train.dropna(subset=[CURRENT["label_col"]])
test = test.dropna(subset=[CURRENT["label_col"]])

# Метки
y_train = train[CURRENT["label_col"]]

# # LR
# search_lr.fit(train["Text"], y_train)
# print("Best params LR:", search_lr.best_params_)
# print("Best F1 macro LR:", search_lr.best_score_)
#
# # XGBoost
# y_train_xgb = y_train - 1
# weights = compute_sample_weight(class_weight='balanced', y=y_train_xgb)
# search_xgb.fit(train["Text"], y_train_xgb, xgb__sample_weight=weights)
#
# print("Best params XGB:", search_xgb.best_params_)
# print("Best F1 macro XGB:", search_xgb.best_score_)


# MLP
start = time.time()
search_mlp.fit(train["Text"], y_train)
mlp_time = time.time() - start
print(f"MLP — Grid Search time: {mlp_time:.2f}s")
print("Best params MLP:", search_mlp.best_params_)
print("Best F1 macro MLP:", search_mlp.best_score_)