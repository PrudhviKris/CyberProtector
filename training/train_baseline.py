"""Train and evaluate the classical ML baseline (leak-free).

Fixes vs. the v1 notebook:
  * no `pd.concat([df, df])` duplication before splitting (that leaked test rows into train)
  * exact duplicates removed after cleaning, then a *stratified* train/test split
  * the vectoriser lives inside an sklearn Pipeline, so inference uses the fitted IDF
  * model selection by 5-fold cross-validation on the training set only

Usage:  python training/train_baseline.py
Outputs (artifacts/): baseline.joblib, metrics.json, model_comparison.png, confusion_matrix.png
"""
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cyberguard.ml.text import clean_text  # noqa: E402

ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)
SEED = 42


def features():
    return FeatureUnion([
        ("word", TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2), min_df=2,
                                 max_features=60000, sublinear_tf=True)),
        ("char", TfidfVectorizer(preprocessor=clean_text, analyzer="char_wb", ngram_range=(3, 5),
                                 min_df=3, max_features=80000, sublinear_tf=True)),
    ])


def load():
    df = pd.read_csv(ROOT / "dataset.csv")
    df["label"] = (df["label"] == -1).astype(int)  # -1 = bullying -> 1, 0 = clean -> 0
    df["clean"] = df["headline"].map(clean_text)
    before = len(df)
    df = df[df["clean"].str.len() > 0].drop_duplicates("clean").reset_index(drop=True)
    print(f"rows: {before} -> {len(df)} after cleaning + de-duplication")
    return df


def main():
    df = load()
    X_tr, X_te, y_tr, y_te = train_test_split(df["headline"], df["label"], test_size=0.2,
                                              stratify=df["label"], random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "LinearSVC": LinearSVC(class_weight="balanced"),
        "ComplementNB": ComplementNB(),
        "SGDClassifier": SGDClassifier(loss="modified_huber", class_weight="balanced", random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight="balanced",
                                               random_state=SEED),
    }
    comparison = []
    for name, clf in candidates.items():
        pipe = Pipeline([("features", features()), ("clf", clf)])
        t0 = time.time()
        cv_f1 = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="f1", n_jobs=-1)
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        row = {
            "model": name,
            "cv_f1_mean": float(cv_f1.mean()), "cv_f1_std": float(cv_f1.std()),
            "test_accuracy": accuracy_score(y_te, pred), "test_precision": precision_score(y_te, pred),
            "test_recall": recall_score(y_te, pred), "test_f1": f1_score(y_te, pred),
            "seconds": round(time.time() - t0, 1),
        }
        comparison.append(row)
        print(f"{name:20s} cvF1={row['cv_f1_mean']:.4f}±{row['cv_f1_std']:.4f}  testF1={row['test_f1']:.4f}")

    # Deploy Logistic Regression (probabilities + fast occlusion explanations), tuned by CV.
    grid = GridSearchCV(Pipeline([("features", features()),
                                  ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))]),
                        {"clf__C": [1, 4, 10, 30]}, cv=cv, scoring="f1", n_jobs=-1)
    grid.fit(X_tr, y_tr)
    best = grid.best_estimator_
    proba = best.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    final = {
        "model": "LogisticRegression (tuned)", "best_params": grid.best_params_,
        "test_accuracy": accuracy_score(y_te, pred), "test_precision": precision_score(y_te, pred),
        "test_recall": recall_score(y_te, pred), "test_f1": f1_score(y_te, pred),
        "test_roc_auc": roc_auc_score(y_te, proba),
        "report": classification_report(y_te, pred, target_names=["non-bullying", "bullying"], output_dict=True),
    }
    print("deployed:", {k: v for k, v in final.items() if k != "report"})

    joblib.dump(best, ART / "baseline.joblib", compress=3)
    meta = {"dataset_rows": int(len(df)), "train_rows": int(len(X_tr)), "test_rows": int(len(X_te)),
            "positive_rate": float(df["label"].mean()), "comparison": comparison, "deployed": final}
    (ART / "metrics.json").write_text(json.dumps(meta, indent=2))

    # plots for the report
    cmp = pd.DataFrame(comparison).sort_values("test_f1")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(cmp["model"], cmp["test_f1"], color="#3b6fd8")
    ax.errorbar(cmp["cv_f1_mean"], cmp["model"], xerr=cmp["cv_f1_std"], fmt="o", color="#d9822b", label="5-fold CV F1")
    ax.set_xlim(max(0, cmp["test_f1"].min() - 0.1), 1)
    ax.set_xlabel("F1 score (bullying class)")
    ax.legend(loc="lower right")
    ax.set_title("Baseline model comparison (held-out test set)")
    fig.tight_layout(); fig.savefig(ART / "model_comparison.png", dpi=150)

    cm = confusion_matrix(y_te, pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["non-bullying", "bullying"]); ax.set_yticks([0, 1], ["non-bullying", "bullying"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Deployed model")
    fig.tight_layout(); fig.savefig(ART / "confusion_matrix.png", dpi=150)
    print("saved artifacts to", ART)


if __name__ == "__main__":
    main()
