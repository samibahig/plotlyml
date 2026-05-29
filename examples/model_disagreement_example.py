"""
Model Disagreement Map — worked example
=========================================
Shows how to feed a real sklearn ensemble and UMAP coordinates.

Dependencies (not required by plotlyml core):
    pip install scikit-learn umap-learn pandas
"""
import numpy as np
from plotlyml import model_disagreement

rng = np.random.default_rng(0)

# ── 1. Synthetic ensemble (no sklearn needed) ─────────────────────────────────
n = 300
# Two-cluster layout in 2-D (e.g. UMAP projection)
cluster = rng.choice([-1, 1], size=n)
x = rng.normal(cluster * 2, 1.5)
y = rng.normal(cluster * 2, 1.5)

# Uncertainty increases near the decision boundary (origin)
dist_to_origin = np.sqrt(x**2 + y**2)
uncertainty = np.clip(1 - dist_to_origin / 3, 0, 1)

true_labels = (cluster > 0).astype(float)
predictions = np.clip(
    true_labels[:, None] + rng.normal(0, uncertainty[:, None] * 0.5 + 0.1, (n, 5)),
    0, 1,
)

fig = model_disagreement(
    x, y,
    predictions,            # shape (n_samples, n_models)
    sample_ids=[f"sample_{i:04d}" for i in range(n)],
    true_labels=true_labels,
    threshold=0.05,
    x_label="UMAP 1",
    y_label="UMAP 2",
    title="Random Forest Ensemble Disagreement — UMAP Projection",
)
fig.show()

# ── 2. Sklearn integration sketch ─────────────────────────────────────────────
# (Uncomment if you have sklearn + umap-learn installed)
#
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# import umap
#
# X, y = make_classification(n_samples=500, n_features=20, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#
# # Train 10 trees individually to get per-tree predictions
# trees = [RandomForestClassifier(n_estimators=1, random_state=i).fit(X_train, y_train)
#          for i in range(10)]
# preds = np.column_stack([t.predict_proba(X_test)[:, 1] for t in trees])
#
# reducer = umap.UMAP(n_components=2, random_state=42)
# emb = reducer.fit_transform(X_test)
#
# fig = model_disagreement(
#     emb[:, 0], emb[:, 1], preds,
#     true_labels=y_test,
#     threshold=0.05,
#     x_label="UMAP 1", y_label="UMAP 2",
# )
# fig.show()
