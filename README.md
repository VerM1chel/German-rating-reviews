# German-rating-reviews

## Results

| Model | Accuracy | F1 macro |
| :--- | :--- | :--- |
| LogisticRegression | 0.7210 | **0.3829** |
| LogisticRegression (Grid Search) | — | 0.3782 |
| XGBoost (200) | 0.8380 | 0.2959 |
| XGBoost (500) | 0.6598 | 0.3513 |
| XGBoost (Grid Search) | — | 0.3490 |

## Experiments

### Baseline
LogisticRegression, max_iter=1000, class_weight='balanced'. F1 macro = 0.3829.

### LogisticRegression (Grid Search)
C, max_features, ngram_range. Лучшие: C=0.66, max_features=10000, ngram_range=(1,2). F1 macro = 0.3782. Не улучшил baseline.

### XGBoost (200 trees)
n_estimators=200, max_depth=6. Без sample_weight. Хуже LR.

### XGBoost (500 trees)
n_estimators=500, max_depth=4. С sample_weight. Лучше, но всё ещё хуже LR.

### XGBoost (Grid Search)
n_estimators, max_depth, max_features. Лучшие: n_estimators=200, max_depth=4, max_features=20000. F1 macro = 0.3490. Хуже LR.

## Выводы
- LogisticRegression лучше по F1 macro.
- Grid Search не улучшил результат.
- XGBoost проигрывает из-за дисбаланса.