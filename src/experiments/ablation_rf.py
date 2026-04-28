import pandas as pd
import numpy as np
import itertools as it
import json
from collections import defaultdict

from sklearn.model_selection import train_test_split, GridSearchCV

from src.utils.pipeline import make_pipeline
from src.utils.metrics import evaluate_model


PARAM_GRID = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [None, 10, 20],
}


def infer_group(colname):
    return colname.split("_")[0] if "_" in colname else colname


def run(data_path):
    df = pd.read_csv(data_path)

    feature_cols = [c for c in df.columns if c not in ["video", "label"]]
    X = df[feature_cols]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipe = make_pipeline()
    gs = GridSearchCV(pipe, PARAM_GRID, cv=3, n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train)

    best_params = gs.best_params_

    scenarios = [{"id": "baseline", "drop": set()}]

    for f in feature_cols:
        scenarios.append({"id": f"lofo__{f}", "drop": {f}})

    results = []

    for s in scenarios:
        used = [c for c in feature_cols if c not in s["drop"]]

        pipe = make_pipeline()
        pipe.set_params(**best_params)

        pipe.fit(X_train[used], y_train)
        metrics = evaluate_model(pipe, X_test[used], y_test)

        results.append({
            "scenario": s["id"],
            **metrics
        })

        print(f"[OK] {s['id']} → acc={metrics['accuracy']:.4f}")

    pd.DataFrame(results).to_csv("results_ablation.csv", index=False)
