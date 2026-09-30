from config import CURRENT
from data_loader import load_data, vectorize
from clean import clean
from search_config import search

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
y_train = train[CURRENT["label_col"]] - 1

# Веса (для дисбаланса)
weights = compute_sample_weight(class_weight='balanced', y=y_train)

# Запуск сетки
search.fit(X_train, y_train, sample_weight=weights)

print("Best params:", search.best_params_)
print("Best F1 macro:", search.best_score_)