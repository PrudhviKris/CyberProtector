"""Optional: fine-tune DistilBERT on this project's own dataset (run on a GPU, e.g. Google Colab).

    pip install transformers datasets accelerate scikit-learn pandas torch
    python training/train_transformer.py --epochs 2 --out artifacts/distilbert-cyberbullying

A fine-tuned model is a binary classifier (label 1 = bullying). To serve it instead of toxic-bert:
    set TRANSFORMER_MODEL=artifacts/distilbert-cyberbullying   (Windows)
    export TRANSFORMER_MODEL=artifacts/distilbert-cyberbullying (Linux/macOS)
The predictor detects a 2-label model, applies softmax, and uses P(label 1) as the transformer score.
Category scores are not available in that mode, because the model only predicts bullying vs. not.
On a T4 GPU, 2 epochs take about 6 minutes. On CPU, expect over an hour.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cyberguard.ml.text import clean_text  # noqa: E402


class DS(torch.utils.data.Dataset):
    def __init__(self, enc, labels):
        self.enc, self.labels = enc, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[i])
        return item


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "distilbert-cyberbullying"))
    args = ap.parse_args()

    df = pd.read_csv(ROOT / "dataset.csv")
    df["label"] = (df["label"] == -1).astype(int)
    df["clean"] = df["headline"].map(clean_text)
    df = df[df["clean"].str.len() > 0].drop_duplicates("clean")
    # same split seed/stratification as the baseline so the numbers are comparable
    X_tr, X_te, y_tr, y_te = train_test_split(df["headline"].tolist(), df["label"].tolist(), test_size=0.2,
                                              stratify=df["label"], random_state=42)

    tok = AutoTokenizer.from_pretrained(args.base)
    enc = lambda xs: tok(xs, truncation=True, max_length=args.max_len)  # noqa: E731
    train_ds, test_ds = DS(enc(X_tr), y_tr), DS(enc(X_te), y_te)
    model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=2)

    def metrics(p):
        logits, labels = p
        prob = torch.softmax(torch.tensor(logits), -1)[:, 1].numpy()
        pred = (prob >= 0.5).astype(int)
        return {"accuracy": accuracy_score(labels, pred), "f1": f1_score(labels, pred),
                "precision": precision_score(labels, pred), "recall": recall_score(labels, pred),
                "roc_auc": roc_auc_score(labels, prob)}

    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir=str(ROOT / "artifacts" / "_hf_runs"), num_train_epochs=args.epochs,
                               per_device_train_batch_size=32, per_device_eval_batch_size=64, learning_rate=3e-5,
                               weight_decay=0.01, warmup_ratio=0.06, eval_strategy="epoch", save_strategy="no",
                               fp16=torch.cuda.is_available(), report_to=[], seed=42),
        train_dataset=train_ds, eval_dataset=test_ds, data_collator=DataCollatorWithPadding(tok),
        compute_metrics=metrics)
    trainer.train()
    result = trainer.evaluate()
    print(json.dumps(result, indent=2))
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    Path(args.out, "eval.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    np.random.seed(42)
    main()
