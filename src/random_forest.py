# ── imports ────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import re

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv("../data/processed/critics_movies.csv", header=0)
# ── example: choose columns ────────────────────────────────────────────────────
num_cols  = ["sentiment"]
target    = "review_score_clean"

# X, y
y = df[target].values
X = df[num_cols]   # text omitted here

# ── train / test split ─────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42)

# ── preprocessing ──────────────────────────────────────────────────────────────
preprocess = ColumnTransformer(
    transformers=[
        ("num",   StandardScaler(), num_cols),   # scale true numeric feature(s)
    ],
    remainder="drop"
)

# ── model ──────────────────────────────────────────────────────────────────────
rf = RandomForestRegressor(
        n_estimators   = 500,
        max_depth      = None,
        min_samples_split = 2,
        min_samples_leaf  = 1,
        n_jobs = -1,
        random_state = 42
)

pipe = Pipeline([
    ("prep", preprocess),
    ("rf",   rf)
])

# ── hyper‑parameter search (randomized, 3‑fold CV) ────────────────────────────
param_dist = {
    "rf__n_estimators"      : [300, 500, 800, 1000],
    "rf__max_depth"         : [None, 10, 20, 40],
    "rf__min_samples_leaf"  : [1, 2, 4],
    "rf__max_features"      : ["sqrt", "log2", 0.25, 0.5],
}

search = RandomizedSearchCV(
        pipe,
        param_distributions = param_dist,
        n_iter       = 25,
        cv           = 3,
        scoring      = "neg_mean_absolute_error",
        n_jobs       = -1,
        verbose      = 1,
        random_state = 42
)

search.fit(X_train, y_train)

print("best params :", search.best_params_)
print("best CV MAE :", -search.best_score_)

# ── final evaluation on held‑out test set ──────────────────────────────────────
best_model = search.best_estimator_
y_pred     = best_model.predict(X_test)


import numpy as np
import matplotlib.pyplot as plt

# ── 1. Scatter: y_true vs y_pred ────────────────────────────────────────────
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         linestyle="--", linewidth=1.5, color="k")
plt.xlabel("True score")
plt.ylabel("Predicted score")
plt.title("Random Forest – Predicted vs. Actual")
plt.tight_layout()
plt.show()

# ── 2. Bar plot: top‑20 feature importances ─────────────────────────────────
# Grab importances + their column names from the pipeline
rf              = pipe.named_steps["rf"]
feature_names   = pipe.named_steps["prep"].get_feature_names_out()
importances     = rf.feature_importances_

# Pick the 20 most important
idx_top         = np.argsort(importances)[-20:]
plt.figure(figsize=(8, 6))
plt.barh(range(len(idx_top)), importances[idx_top], align="center")
plt.yticks(range(len(idx_top)), feature_names[idx_top])
plt.xlabel("Gini importance")
plt.title("Top‑20 Feature Importances\n(Random Forest)")
plt.tight_layout()
plt.show()


print("Test MAE :", mean_absolute_error(y_test, y_pred))
print("Test R²  :", r2_score(y_test, y_pred))
