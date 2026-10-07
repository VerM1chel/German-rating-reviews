# German-rating-reviews

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Download the dataset from Kaggle and place it in the project root.
3. Train the baseline: `cd core && python ml_train.py`
4. Train DistilBERT: `python -m distilbert.train`
5. Experiments: `experiments/*.ipynb`

## Data

- **Source:** [Patient Reviews of Doctor's (Kaggle)](https://www.kaggle.com/datasets/thedevastator/german-2021-patient-reviews-and-ratings-of-docto)
- **License:** CC BY-SA 4.0
- **Volume:** 439,280 reviews
- **Missing `rating`:** 9,544 (2%) — dropped
- **Class imbalance:** 81% — rating 1, remaining 19% — ratings 2–6
- **Mean rating:** 1.56 (Std = 1.34)
- **Split:** 80/20, `random_state=42`, no stratification (verified: class proportion difference < 0.001)


## Results
| Model | Accuracy | F1 macro | Train time | Inference time |
| :--- | :--- |:---------| :--- | :--- |
| LogisticRegression | 0.7055 | 0.3786   | 41s | 0.01s |
| LogisticRegression (Grid Search) | — | 0.3782   | — | — |
| LogisticRegression (alternative preprocessing) | 0.7057 | 0.3798   | 34s | 0.02s |
| MLPClassifier | 0.8273 | 0.3737   | 4h | 0.55s |
| XGBoost (200) | 0.8380 | 0.2959   | — | — |
| XGBoost (500) | 0.6598 | 0.3513   | — | — |
| XGBoost (Grid Search) | — | 0.3490   | — | — |
| XGBoost (train.py) | 0.6478 | 0.3404   | 1.9h | 1.32s |
| SVM (linear kernel) | 0.8188 | 0.3855   | 38s | — |
| SVM (RBF, 20k) | 0.8370 | 0.3318   | 80s | — |
| LSTM (80k, 14M) | 0.7451 | 0.3872   | 1072s | 10.49s |
| LSTM (160k, 28M) | 0.7300 | 0.3867   | 7051s | 89.42s |
| DistilBERT (GTX 1650 GPU) | 0.8608 | 0.4405   | 4h | 9 min |


## Per-class F1 (LogisticRegression)
| Class | Support | Precision | Recall | F1 |
| :--- | ---: | ---: | ---: | ---: |
| 1 | 70108 | 0.971 | 0.779 | 0.864 |
| 2 | 4482 | 0.127 | 0.408 | 0.193 |
| 3 | 1633 | 0.139 | 0.323 | 0.194 |
| 4 | 2430 | 0.185 | 0.271 | 0.220 |
| 5 | 3930 | 0.339 | 0.331 | 0.335 |
| 6 | 3389 | 0.423 | 0.516 | 0.465 |

F1 macro = 0.379 hides the unevenness: 0.86 on class 1 and 0.19–0.46 
on classes 2–6. Quality drops with decreasing class size, but not only — 
class 2 (4482 samples) yields a lower F1 than class 4 (2430 samples), which 
suggests textual proximity between ratings 1 and 2.

## Per-class F1 comparison

Three models were selected — one per approach: 
LR (linear + TF-IDF), LSTM (RNN), DistilBERT (transformer). 
Other models (MLP, XGBoost, SVM) showed results close to LR 
at the macro level and provide no new information at the per-class level.

| Class | LR | LSTM | DistilBERT |
| :--- | ---: | ---: | ---: |
| 1 | 0.864 | 0.893 | 0.963 |
| 2 | 0.193 | 0.202 | 0.211 |
| 3 | 0.194 | 0.217 | 0.275 |
| 4 | 0.220 | 0.258 | 0.238 |
| 5 | 0.335 | 0.355 | 0.452 |
| 6 | 0.465 | 0.399 | 0.504 |
| Macro | 0.379 | 0.387 | 0.441 |

## Experiments

### Baseline
LogisticRegression, max_iter=1000, class_weight='balanced'. F1 macro = 0.3786.

### LogisticRegression (Grid Search)
C, max_features, ngram_range. Best: C=0.66, max_features=10000, ngram_range=(1,2). F1 macro = 0.3782. Did not improve the baseline.

### XGBoost (200 trees)
n_estimators=200, max_depth=6. Without sample_weight. Worse than LR.

### XGBoost (500 trees)
n_estimators=500, max_depth=4. With sample_weight. Better, but still worse than LR.

### XGBoost (Grid Search)
n_estimators, max_depth, max_features. Best: n_estimators=200, max_depth=4, max_features=20000. F1 macro = 0.3490. Worse than LR.

### LR + ngram(1,2) + sublinear_tf
F1 macro = 0.3829 vs 0.3786 for the baseline. Gain +0.0043 — within noise. 
Bigrams and sublinear_tf provide no significant improvement: TF-IDF is already 
at its limit for a linear model.

### SVM with linear kernel (LinearSVC) on the full dataset
F1 macro = 0.3855 vs 0.379 for LR. Gain +0.007 — within noise. 
A different classification algorithm provides no significant improvement: 
the problem is not in the choice of a linear model.

### SVM (RBF kernel) on a 20k subset
F1 macro = 0.332 vs 0.379 for LR on the full dataset. The non-linearity 
hypothesis is not confirmed: the RBF kernel on sparse TF-IDF features 
collapses to the dominant class (class 1: F1 = 0.94, class 3: F1 = 0.02).

### LSTM: increasing data and model size does not help
- LSTM v1 (80k train, 14M params): F1 macro = 0.3872
- LSTM v2 (160k train, 28M params): F1 macro = 0.3867

Twice the data and parameters — the result did not change (0.3872 vs 0.3867, 
the difference is within noise). This confirms: LSTM is limited not by data volume, 
but by architecture. Sequential reading provides no advantage over TF-IDF + LR 
on short texts with imbalance and close classes.

### DistilBERT (distilbert-base-german-cased, 80k, 2 epochs)
F1 macro = 0.4405 vs 0.379 for LR and 0.387 for LSTM. Gain +0.054 — significant.

Per-class F1 shows: DistilBERT better distinguishes pronounced ratings 
(class 1: 0.963 vs 0.864; class 5: 0.452 vs 0.335; class 6: 0.504 vs 0.465), 
but on gradations 2–4 the gain is minimal (0.21–0.27 vs 0.19–0.22 for LR). 
The transformer pulls up classes with a clear textual signal, but does not solve 
the problem of textual proximity between adjacent ratings.

### Verified — no changes needed
- Stratification at split: verified, the difference in class proportions between 
  train and test < 0.001. On a 439k dataset it does not affect the metric.

### Room for improvement
- Train MLP with class_weight to make the comparison with LogisticRegression fair.
- Try merging rare classes (3 classes: {1}, {2,3,4}, {5,6}) — 
  to check where exactly the model breaks.

## Conclusions

### LogisticRegression — best quality/time trade-off
LogisticRegression yields F1 macro = 0.379 with 41 seconds of training. 
MLPClassifier shows almost the same result (0.374) but takes 4 hours to train. 
With limited time, LogisticRegression is the obvious choice.

### Binary setting (1 vs not-1)
F1 macro = 0.837 vs 0.379 in the 6-class setting.
Train time: 1.5s vs 41s.

Conclusion: the linear model confidently separates negative from non-negative, 
but does not distinguish gradations of ratings (2–6). The problem of the 6-class 
task is not in the model or the features, but in excessive detail.
Accurate rating 1–6 requires a model with context (DistilBERT).

### Quality depends on class size and proximity
The model separates rating 1 from the rest well (F1 = 0.86), but within 
"not-1" it performs at the level of random guessing (F1 = 0.19–0.46). 
This is not a defect of the model, but a property of the data: strong imbalance 
(81% — rating 1) and textual proximity between adjacent ratings (1-2, 5-6).

### XGBoost loses even with class balancing
XGBoost with imbalance compensation (sample_weight) yields F1 macro = 0.351 — 
still lower than LogisticRegression (0.379). So the issue is not only 
the imbalance. The likely reason — gradient boosting performs worse 
with sparse TF-IDF features in a 6-class task. This is a hypothesis 
that requires separate verification.

### Grid Search did not improve the baseline
Hyperparameter search for LogisticRegression yielded F1 macro = 0.378 
vs 0.379 for the baseline. The reason: the search was conducted simultaneously 
over hyperparameters and over the feature space, so the comparison with the baseline 
is not entirely fair. To honestly assess the contribution of hyperparameters, 
the features must be fixed.

### Preprocessing impact
The first version of preprocessing removed negations (nicht, kein) as stopwords 
from NLTK and collapsed any letter repetitions (dass → das, bitte → bite). This 
could potentially distort the signal for classes 2–6.

After the fix (negations preserved, repetitions collapsed only for 3+): 
F1 macro = 0.3798 vs 0.3786. Gain +0.001 — within noise.

Conclusion: preprocessing was not the bottleneck. Even with correct processing, 
LR hits the same ceiling of ~0.38. However, the fix is important 
for reproducibility and pipeline correctness.

### DistilBERT — significant gain, but not a solution
DistilBERT yields F1 macro = 0.441 vs 0.379 for LR. Gain +0.054 — 
the first significant result across all experiments. However, per-class F1 
shows that the improvement is concentrated on classes with a clear signal (1, 5, 6); 
gradations 2–4 remain difficult for any model. The ceiling of the task is 
not in the model, but in the data: subjectivity of ratings and textual proximity 
of reviews with ratings 2, 3, 4.

### Classical ML is exhausted
Six models (LR, XGBoost, MLP, SVM linear, SVM RBF, binary LR) yield 
F1 macro in the range 0.33–0.39 on the 6-class task.

Hyperparameters: tuned for LR and XGBoost (Grid Search). For MLP and SVM — 
not tuned, because the baseline already showed no significant gain, and CPU 
resources are limited. The spread of results between models (0.33–0.39) is too small 
for tuning one of them to dramatically change the picture.

Features: ngram(1,2) and sublinear_tf on LR. Gain +0.004 — within noise.
Preprocessing: preserving negations on LR. Gain +0.001 — within noise.

None of the factors provides a significant gain. The reason — data structure: 
strong imbalance (81% — rating 1) and textual proximity of ratings 2–6. 
The 6-class task requires a model with context (DistilBERT), 
which yields F1 macro = 0.441 — a significant, but not radical gain.

## Model
DistilBERT model is available on HuggingFace Hub: [VerMichel/german-reviews-distilbert](https://huggingface.co/VerMichel/german-reviews-distilbert)