"""Fit and freeze a pair classifier on train/dev, or score a frozen split."""

import argparse
from difflib import SequenceMatcher
from hashlib import sha256
from itertools import chain
import json
from pathlib import Path
import re
import unicodedata

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples/experiments/pair-paraphrase-v1"
FREEZE = ROOT / ".local/pair-paraphrase/baseline-freeze.json"
FILES = {"train": "train.jsonl", "dev": "dev.jsonl",
         "producer": "producer_test.jsonl", "receiver": "receiver_test.jsonl"}
SEED = 20261109
THRESHOLD = 0.5
NEGATIONS = frozenset(("no", "not", "never", "neither", "nor", "without", "none", "cannot"))
NUMBERS = re.compile(r"\d+(?:[.,:/-]\d+)*")
WORDS = re.compile(r"[^\W_]+(?:'[^\W_]+)?")
VERSION = "pair-paraphrase-v1"


def sha(data):
    return sha256(data).hexdigest()


def read_split(data_dir, split):
    raw = (data_dir / FILES[split]).read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    if not rows or any(row["label"] not in ("same", "different") for row in rows):
        raise ValueError(f"Empty split or invalid label in {split}")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"Repeated ID in {split}")
    return rows, sha(raw)


def pairs(rows):
    for row in rows:
        pair = json.loads(row["text"])
        # Canonicalizing the order also makes SequenceMatcher's tie behavior symmetric.
        yield tuple(sorted(
            unicodedata.normalize("NFKC", pair[key]).casefold() for key in ("a", "b")
        ))


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


def negations(tokens):
    return {word for word in tokens if word in NEGATIONS or word.endswith("n't")}


def pair_features(a, b):
    x, y = WORDS.findall(a), WORDS.findall(b)
    xnum, ynum = set(NUMBERS.findall(a)), set(NUMBERS.findall(b))
    xneg, yneg = negations(x), negations(y)
    order = SequenceMatcher(None, x, y, autojunk=False).ratio()
    lexical = jaccard(set(x), set(y))
    return [
        lexical,
        jaccard(set(zip(x, x[1:])), set(zip(y, y[1:]))),
        min(len(x), len(y)) / max(1, len(x), len(y)),
        order,
        lexical - order,
        SequenceMatcher(None, a, b, autojunk=False).ratio(),
        float(xnum == ynum),
        jaccard(xnum, ynum),
        float(xneg == yneg),
        float(bool(xneg) != bool(yneg)),
    ]


def fit_vectorizers(training_pairs):
    sentences = list(chain.from_iterable(training_pairs))
    return [
        TfidfVectorizer(analyzer=analyzer, ngram_range=ngrams, min_df=2, sublinear_tf=True)
        .fit(sentences)
        for analyzer, ngrams in [("word", (1, 2)), ("char_wb", (2, 5))]
    ]


def features(vectorizers, input_pairs):
    a, b = zip(*input_pairs)
    similarities = [
        np.asarray(vectorizer.transform(a).multiply(vectorizer.transform(b)).sum(axis=1)).ravel()
        for vectorizer in vectorizers
    ]
    return np.column_stack([*similarities, np.array([pair_features(x, y) for x, y in input_pairs])])


def candidates():
    return {
        "tfidf_pair_logistic": make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, max_iter=500, random_state=SEED)
        ),
        "tfidf_pair_hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.08, max_iter=120, max_leaf_nodes=15, min_samples_leaf=16,
            l2_regularization=1.0, random_state=SEED,
        ),
    }


def evaluate(model, matrix, rows):
    with threadpool_limits(limits=1):
        scores = model.predict_proba(matrix)[:, 1]
    predictions = []
    by_label = {label: {"total": 0, "correct": 0} for label in ("same", "different")}
    for row, score in zip(rows, scores):
        prediction = "same" if score >= THRESHOLD else "different"
        expected = row["label"]
        by_label[expected]["total"] += 1
        by_label[expected]["correct"] += prediction == expected
        predictions.append({"id": row["id"], "label": prediction, "score_same": float(score)})
    negatives = by_label["different"]
    false_same = negatives["total"] - negatives["correct"]
    return {
        "total": len(rows),
        "correct": sum(count["correct"] for count in by_label.values()),
        "false_same": false_same,
        "false_same_rate": false_same / negatives["total"] if negatives["total"] else None,
        "by_label": by_label,
        "predictions": predictions,
    }


def fit(training, development):
    train_pairs = list(pairs(training))
    vectorizers = fit_vectorizers(train_pairs)
    train_matrix = features(vectorizers, train_pairs)
    dev_matrix = features(vectorizers, list(pairs(development)))
    targets = [row["label"] == "same" for row in training]
    trained = candidates()
    outcomes = {}
    for name, model in trained.items():
        with threadpool_limits(limits=1):
            model.fit(train_matrix, targets)
        outcomes[name] = evaluate(model, dev_matrix, development)
    winner = max(outcomes, key=lambda name: (outcomes[name]["correct"], -outcomes[name]["false_same"]))
    return winner, (vectorizers, trained[winner]), outcomes


def freeze(data_dir, freeze_path):
    model_path = freeze_path.with_suffix(".joblib")
    if freeze_path.exists() or model_path.exists():
        raise ValueError("Choose a new path for the CPU freeze")
    training, training_hash = read_split(data_dir, "train")
    development, development_hash = read_split(data_dir, "dev")
    winner, model, outcomes = fit(training, development)
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("xb") as output:
        joblib.dump(model, output, compress=3)
    record = {
        "version": VERSION,
        "baseline_name": winner,
        "baseline_model_sha256": sha(model_path.read_bytes()),
        "source_sha256": sha(Path(__file__).read_bytes()),
        "training_sha256": training_hash,
        "development_sha256": development_hash,
        "sklearn_version": sklearn.__version__,
        "threshold": THRESHOLD,
        "development": outcomes,
    }
    with freeze_path.open("x", encoding="utf-8") as output:
        json.dump(record, output, indent=2)
        output.write("\n")
    return record


def score(data_dir, freeze_path, split):
    record = json.loads(freeze_path.read_bytes())
    if (record["version"] != VERSION or record["threshold"] != THRESHOLD
            or record["source_sha256"] != sha(Path(__file__).read_bytes())
            or record["sklearn_version"] != sklearn.__version__):
        raise ValueError("The CPU freeze does not match this source or environment")
    model_path = freeze_path.with_suffix(".joblib")
    with model_path.open("rb") as model_file:
        if sha(model_file.read()) != record["baseline_model_sha256"]:
            raise ValueError("The CPU model does not match its local freeze digest")
        model_file.seek(0)
        # This artifact is produced locally by --freeze and must come from a trusted path.
        vectorizers, model = joblib.load(model_file)
    rows, holdout_hash = read_split(data_dir, split)
    return {
        "ok": True,
        "split": split,
        "holdout_sha256": holdout_hash,
        **{key: record[key] for key in (
            "baseline_name", "baseline_model_sha256", "source_sha256", "training_sha256",
        )},
        **evaluate(model, features(vectorizers, list(pairs(rows))), rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_mutually_exclusive_group(required=True)
    command.add_argument("--freeze", action="store_true")
    command.add_argument("--split", choices=("producer", "receiver"))
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--freeze-path", type=Path, default=FREEZE)
    args = parser.parse_args()
    if args.freeze:
        record = freeze(args.data_dir, args.freeze_path)
        print(json.dumps({
            "ok": True,
            "baseline_name": record["baseline_name"],
            "freeze_path": str(args.freeze_path),
            "development": {
                name: {key: result[key] for key in ("correct", "total", "false_same")}
                for name, result in record["development"].items()
            },
        }))
    else:
        print(json.dumps(score(args.data_dir, args.freeze_path, args.split)))


if __name__ == "__main__":
    main()
