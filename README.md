# German-rating-reviews

## Results

| Model | Accuracy | F1 macro |
| :--- | :--- | :--- |
| LogisticRegression | 0.7210 | 0.3829 |
| XGBoost (200) | 0.8380 | 0.2959 |
| XGBoost (500) | 0.6598 | 0.3513 |

## Experiments

### Baseline
LogisticRegression, max_iter=1000, class_weight='balanced'.

### XGBoost (200 trees)
n_estimators=200, max_depth=6. Без sample_weight. Хуже LR.

### XGBoost (500 trees)
n_estimators=500, max_depth=4. С sample_weight. Лучше, но всё ещё хуже LR.

## Выводы
- LogisticRegression лучше по F1 macro.
- XGBoost проигрывает из-за дисбаланса.