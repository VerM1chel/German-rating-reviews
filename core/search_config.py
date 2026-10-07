from core.models import get_mlp

from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from scipy.stats import uniform
from nltk.corpus import stopwords

pipeline_lr = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=stopwords.words('german'))),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
])

param_dist_lr = {
    'tfidf__max_features': [5000, 10000, 20000],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'clf__C': uniform(0.1, 10),
}

search_lr = RandomizedSearchCV(
    pipeline_lr,
    param_dist_lr,
    n_iter=10,
    cv=3,
    scoring='f1_macro',
    random_state=42,
    verbose=1
)

pipeline_xgb = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=stopwords.words('german'))),
    ('xgb', XGBClassifier(random_state=42, eval_metric='mlogloss'))
])

param_dist_xgb = {
    'tfidf__max_features': [5000, 10000, 20000],
    'xgb__n_estimators': [100, 200],
    'xgb__max_depth': [4, 6],
}

search_xgb = RandomizedSearchCV(
    pipeline_xgb,
    param_dist_xgb,
    n_iter=4,
    cv=2,
    scoring='f1_macro',
    random_state=42,
    verbose=1
)

# MLP
pipeline_mlp = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=stopwords.words('german'))),
    ('clf', get_mlp())
])

search_mlp = RandomizedSearchCV(
    pipeline_mlp,
    {'clf__hidden_layer_sizes': [(50,), (100,), (100, 50)]},
    n_iter=10,
    cv=3,
    scoring='f1_macro',
    random_state=42,
    verbose=1
)