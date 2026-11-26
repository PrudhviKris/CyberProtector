"""Inference service: classical baseline + pretrained transformer ensemble with explanations."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np

from .text import tokenize_display

log = logging.getLogger(__name__)

CATEGORY_LABELS = {
    "toxic": "Toxic",
    "severe_toxic": "Severe toxicity",
    "obscene": "Obscene",
    "threat": "Threat",
    "insult": "Insult",
    "identity_hate": "Identity hate",
}


@dataclass
class Analysis:
    text: str
    label: str                      # "bullying" | "non-bullying"
    score: float                    # final probability of bullying (0-1)
    severity: str                   # none | low | medium | high
    baseline_score: float
    transformer_score: float | None
    categories: dict = field(default_factory=dict)    # category -> probability
    top_category: str | None = None
    highlights: list = field(default_factory=list)    # [{"token", "weight"}]
    models: list = field(default_factory=list)

    @property
    def is_bullying(self) -> bool:
        return self.label == "bullying"

    def to_dict(self) -> dict:
        return {
            "text": self.text, "label": self.label, "is_bullying": self.is_bullying,
            "score": round(self.score, 4), "severity": self.severity,
            "baseline_score": round(self.baseline_score, 4),
            "transformer_score": None if self.transformer_score is None else round(self.transformer_score, 4),
            "categories": {k: round(v, 4) for k, v in self.categories.items()},
            "top_category": self.top_category, "highlights": self.highlights, "models": self.models,
        }


class Predictor:
    def __init__(self, baseline_path: str | Path, transformer_name: str | None = "unitary/toxic-bert",
                 threshold: float = 0.5, baseline_weight: float = 0.4, max_chars: int = 5000):
        self.baseline = joblib.load(baseline_path)
        self.transformer_name = transformer_name
        self.threshold = threshold
        self.baseline_weight = baseline_weight
        self.max_chars = max_chars
        self._tok = self._model = None
        self._lock = threading.Lock()
        self._transformer_failed = False

    # ---------- transformer (lazy) ----------
    def _load_transformer(self) -> bool:
        if not self.transformer_name or self._transformer_failed:
            return False
        if self._model is not None:
            return True
        with self._lock:
            if self._model is None:
                try:
                    import torch  # noqa: F401
                    from transformers import AutoModelForSequenceClassification, AutoTokenizer
                    self._tok = AutoTokenizer.from_pretrained(self.transformer_name)
                    self._model = AutoModelForSequenceClassification.from_pretrained(self.transformer_name).eval()
                    log.info("Loaded transformer %s", self.transformer_name)
                except Exception as exc:  # offline / not installed -> baseline only
                    log.warning("Transformer unavailable (%s); using baseline only", exc)
                    self._transformer_failed = True
                    return False
        return True

    @property
    def transformer_ready(self) -> bool:
        return self._model is not None

    def warmup(self):
        self._load_transformer()

    def _transformer_scores(self, texts: list[str]) -> list[dict] | None:
        if not self._load_transformer():
            return None
        import torch
        out = []
        cfg = self._model.config
        # toxic-bert is multi-label (sigmoid per label); our own fine-tuned model is binary (softmax)
        multilabel = cfg.problem_type == "multi_label_classification" or cfg.num_labels > 2
        for i in range(0, len(texts), 16):
            enc = self._tok(texts[i:i + 16], truncation=True, max_length=256, padding=True, return_tensors="pt")
            with torch.no_grad():
                logits = self._model(**enc).logits
            if multilabel:
                for row in torch.sigmoid(logits).numpy():
                    out.append({cfg.id2label[j]: float(p) for j, p in enumerate(row)})
            else:
                for p in torch.softmax(logits, -1)[:, 1].numpy():
                    out.append({"toxic": float(p)})
        return out

    # ---------- explanation ----------
    def _explain(self, text: str, limit: int = 60) -> list[dict]:
        """Occlusion: drop each distinct word and measure the change in the model's log-odds.

        Log-odds (not probability) are used so the signal survives when the probability saturates
        near 0 or 1. Weights are normalised to [-1, 1] relative to the most influential word;
        positive = pushes toward bullying.
        """
        tokens = tokenize_display(text)
        uniq = list(dict.fromkeys(t.lower() for t in tokens))[:limit]
        if not uniq:
            return []
        variants = [" ".join(t for t in tokens if t.lower() != u) for u in uniq]
        logits = self.baseline.decision_function([text] + variants)
        deltas = logits[0] - logits[1:]
        peak = float(np.max(np.abs(deltas))) or 1.0
        res = [{"token": u, "weight": round(float(d / peak), 4)} for u, d in zip(uniq, deltas)
               if abs(d / peak) >= 0.1 and abs(d) >= 0.05]
        return sorted(res, key=lambda r: -r["weight"])

    # ---------- public API ----------
    def analyze_many(self, texts: list[str], explain: bool = True) -> list[Analysis]:
        texts = [(t or "")[: self.max_chars] for t in texts]
        base = self.baseline.predict_proba(texts)[:, 1]
        trans = self._transformer_scores(texts)
        results = []
        for i, text in enumerate(texts):
            b = float(base[i])
            cats, t_score, models = {}, None, ["tfidf-logreg"]
            if trans is not None:
                cats = trans[i]
                t_score = max(cats.values())
                models.append(self.transformer_name)
                score = self.baseline_weight * b + (1 - self.baseline_weight) * t_score
            else:
                score = b
            label = "bullying" if score >= self.threshold else "non-bullying"
            if label == "non-bullying":
                severity = "none"
            elif score >= 0.85 or cats.get("severe_toxic", 0) >= 0.3 or cats.get("threat", 0) >= 0.5:
                severity = "high"
            elif score >= 0.7:
                severity = "medium"
            else:
                severity = "low"
            top = None
            if cats and label == "bullying":
                specific = {k: v for k, v in cats.items() if k != "toxic"}
                k = max(specific, key=specific.get)
                top = k if specific[k] >= 0.3 else "toxic"
            results.append(Analysis(
                text=text, label=label, score=float(score), severity=severity, baseline_score=b,
                transformer_score=t_score, categories=cats, top_category=top,
                highlights=self._explain(text) if explain and text.strip() else [], models=models))
        return results

    def analyze(self, text: str, explain: bool = True) -> Analysis:
        return self.analyze_many([text], explain=explain)[0]


def highlight_spans(text: str, highlights: list[dict]) -> list[tuple[str, float]]:
    """Split text into (segment, weight) pairs for rendering highlighted words."""
    import re
    weights = {h["token"]: h["weight"] for h in highlights if h["weight"] > 0}
    if not weights:
        return [(text, 0.0)]
    parts = re.split(r"([A-Za-z0-9@$*']+)", text)
    return [(p, weights.get(p.lower(), 0.0)) for p in parts if p]


def safe_mean(xs):
    return float(np.mean(xs)) if len(xs) else 0.0
