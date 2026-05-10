from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "weapons_dataset_normalized.csv"
DEFAULT_SUPPLEMENTAL_DATASET = ROOT / "data" / "curated_visual_descriptions.csv"
DEFAULT_ARTIFACT_DIR = ROOT / "artifacts"
DEFAULT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# ----------------------------
# Data Loading
# ----------------------------
def load_weapon_dataset(dataset_path: Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)

    if DEFAULT_SUPPLEMENTAL_DATASET.exists():
        df = pd.concat([df, pd.read_csv(DEFAULT_SUPPLEMENTAL_DATASET)], ignore_index=True)

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df = df[(df["text"] != "") & (df["label"] != "")]

    if df.empty:
        raise ValueError("Dataset is empty after cleaning.")

    return df


# ----------------------------
# Metrics
# ----------------------------
def top_k_accuracy(y_true, probabilities, classes, k=3):
    top_k_idx = np.argsort(probabilities, axis=1)[:, -k:]
    top_labels = np.take(classes, top_k_idx)
    return float(np.mean([t in preds for t, preds in zip(y_true, top_labels)]))

def mean_reciprocal_rank(y_true, probabilities, classes):
    ranks = []

    for truth, probs in zip(y_true, probabilities):
        ranked = classes[np.argsort(probs)[::-1]]
        rank = np.where(ranked == truth)[0][0] + 1
        ranks.append(1 / rank)

    return float(np.mean(ranks))

def evaluate_model(name, model, X_test, y_test, classes):
    preds = model.predict(X_test)

    # Some models (SVM) don't support predict_proba
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)
    else:
        # fallback: fake probabilities
        probs = np.eye(len(classes))[model.predict(X_test)]

    baseline_preds = cosine_baseline(X_train, y_train, X_test)

    baseline_acc = accuracy_score(y_test, baseline_preds)
    print("Baseline kNN-like accuracy:", baseline_acc)

    return {
        "model": name,
        "accuracy": float(accuracy_score(y_test, preds)),
        "top_3_accuracy": top_k_accuracy(y_test, probs, classes, k=min(3, len(classes))),
        "report": classification_report(y_test, preds, output_dict=True, zero_division=0),
    }


# ----------------------------
# Training
# ----------------------------
def train_model(
    dataset_path: Path,
    artifact_dir: Path,
    model_name: str,
    test_size: float = 0.2,
    random_state: int = 42,
):
    df = load_weapon_dataset(dataset_path)

    # ----------------------------
    # Proper split (stratified, not fake grouping)
    # ----------------------------
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state,
    )

    embedder = SentenceTransformer(model_name)

    X_train = embedder.encode(
        train_df["text"].tolist(),
        normalize_embeddings=False,
        show_progress_bar=True,
    )

    X_test = embedder.encode(
        test_df["text"].tolist(),
        normalize_embeddings=False,
        show_progress_bar=True,
    )

    y_train = train_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    classes = np.unique(y_train)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # ----------------------------
    # Models
    # ----------------------------
    models = {
        "logreg": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
        ),
        "svm": SGDClassifier(loss="log_loss", max_iter=2000),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
        ),
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"Training {name}...")

        model.fit(X_train, y_train)
        trained_models[name] = model

        preds = model.predict(X_test)

        # ----------------------------
        # Safe probability handling
        # ----------------------------
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)
        else:
            # convert hard predictions → one-hot pseudo-probabilities
            probs = np.zeros((len(preds), len(classes)))
            class_to_idx = {c: i for i, c in enumerate(classes)}

            for i, p in enumerate(preds):
                probs[i, class_to_idx[p]] = 1.0

        result = {
            "model": name,
            "accuracy": float(accuracy_score(y_test, preds)),
            "f1_macro": float(f1_score(y_test, preds, average="macro")),
            "top_3_accuracy": top_k_accuracy(
                y_test, probs, classes, k=min(3, len(classes))
            ),
            "mrr": mean_reciprocal_rank(y_test, probs, classes),
            "report": classification_report(
                y_test, preds,
                output_dict=True,
                zero_division=0
            ),
        }

        results.append(result)

    # ----------------------------
    # Pick best model
    # ----------------------------
    best = max(results, key=lambda x: x["accuracy"])
    best_model = trained_models[best["model"]]

    print("\nBest model:", best["model"])

    # ----------------------------
    # Save artifacts
    # ----------------------------
    artifact_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, artifact_dir / "classifier.joblib")

    (artifact_dir / "model_config.json").write_text(
        json.dumps(
            {
                "embedding_model": model_name,
                "labels": classes.tolist(),
                "best_model": best["model"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (artifact_dir / "metrics.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    return results

def cosine_baseline(X_train, y_train, X_test):
    sims = X_test @ X_train.T
    preds = []

    for row in sims:
        idx = np.argmax(row)
        preds.append(y_train[idx])

    return preds

# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)

    args = parser.parse_args()

    results = train_model(args.dataset, args.artifacts, args.model_name)

    print("\n=== MODEL COMPARISON ===")
    for r in results:
        print(f"{r['model']}: acc={r['accuracy']:.3f}, top3={r['top_3_accuracy']:.3f}")


if __name__ == "__main__":
    main()