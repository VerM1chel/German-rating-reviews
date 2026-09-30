from models import get_xgb
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(3, 8),
}

search = RandomizedSearchCV(
    get_xgb(),
    param_dist,
    n_iter=10,
    cv=3,
    scoring='f1_macro',
    random_state=42,
    verbose=1
)