"""Train a lightweight learned MEPRS reranker.

This revision experiment treats the existing LLM reviewer scores as features and
learns a small neural aggregator that predicts candidate BLEURT. Evaluation is
leave-one language-direction/strategy out: for each held-out condition, the
model is trained on the remaining released MEPRS data and then selects one of
the five candidate translations for every held-out sentence.

The script does not call any LLM APIs and does not require source/reference
alignment. It is meant as a low-cost new experiment that can use a GPU when a
compatible PyTorch build is available.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


MODEL_CODES = ["G35", "G4o", "C3", "C35", "GP"]
STRATEGIES = ["it", "tear"]
DIMENSIONS = ["score", "accuracy", "fluency", "style", "terminology"]


@dataclass
class ExampleSet:
    pair: str
    strategy: str
    x: np.ndarray
    y: np.ndarray
    sentence_idx: np.ndarray
    candidate_idx: np.ndarray
    candidate_bleurt: np.ndarray
    meprs_avg_bleurt: float
    best_baseline_bleurt: float
    oracle_bleurt: float


def read_scores(path: Path) -> np.ndarray:
    values = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                values.append(float(line.split()[0]))
    return np.asarray(values, dtype=np.float32)


def safe_cuda_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        tensor = torch.tensor([1.0], device="cuda")
        _ = tensor * 2.0
        torch.cuda.synchronize()
        return True
    except Exception:
        return False


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" and safe_cuda_available():
        return torch.device("cuda")
    if requested == "cuda":
        return torch.device("cpu")
    return torch.device("cuda" if safe_cuda_available() else "cpu")


def load_condition(pair_dir: Path, strategy: str) -> ExampleSet:
    pair = pair_dir.name.replace("-new", "")
    metric = np.vstack(
        [read_scores(pair_dir / f"{candidate}_{strategy}.bleurt") for candidate in MODEL_CODES]
    )
    n_sent = metric.shape[1]

    # reviewer_scores[candidate, reviewer, dimension, sentence]
    reviewer_scores = np.zeros(
        (len(MODEL_CODES), len(MODEL_CODES), len(DIMENSIONS), n_sent), dtype=np.float32
    )
    for candidate_i, candidate in enumerate(MODEL_CODES):
        for reviewer_i, reviewer in enumerate(MODEL_CODES):
            for dim_i, dim in enumerate(DIMENSIONS):
                reviewer_scores[candidate_i, reviewer_i, dim_i, :] = read_scores(
                    pair_dir / f"{candidate}_{strategy}_{reviewer}.{dim}"
                )

    rows = []
    labels = []
    sentence_idx = []
    candidate_idx = []
    for sent in range(n_sent):
        for candidate_i in range(len(MODEL_CODES)):
            raw = reviewer_scores[candidate_i, :, :, sent].reshape(-1)
            by_dim = reviewer_scores[candidate_i, :, :, sent]
            summary = np.concatenate(
                [
                    by_dim.mean(axis=0),
                    by_dim.std(axis=0),
                    by_dim.min(axis=0),
                    by_dim.max(axis=0),
                ]
            )
            candidate_one_hot = np.eye(len(MODEL_CODES), dtype=np.float32)[candidate_i]
            strategy_one_hot = np.asarray([strategy == "it", strategy == "tear"], dtype=np.float32)
            rows.append(np.concatenate([raw, summary, candidate_one_hot, strategy_one_hot]))
            labels.append(metric[candidate_i, sent])
            sentence_idx.append(sent)
            candidate_idx.append(candidate_i)

    mean_score_selection = reviewer_scores[:, :, 0, :].sum(axis=1)
    meprs_selected = np.argmax(mean_score_selection, axis=0)
    meprs_scores = metric[meprs_selected, np.arange(n_sent)]

    return ExampleSet(
        pair=pair,
        strategy=strategy,
        x=np.vstack(rows).astype(np.float32),
        y=np.asarray(labels, dtype=np.float32),
        sentence_idx=np.asarray(sentence_idx, dtype=np.int64),
        candidate_idx=np.asarray(candidate_idx, dtype=np.int64),
        candidate_bleurt=metric,
        meprs_avg_bleurt=float(meprs_scores.mean() * 100),
        best_baseline_bleurt=float(metric.mean(axis=1).max() * 100),
        oracle_bleurt=float(metric.max(axis=0).mean() * 100),
    )


class Reranker(nn.Module):
    def __init__(self, input_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, max(8, hidden // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(8, hidden // 2), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_predict(
    train_sets: list[ExampleSet],
    test_set: ExampleSet,
    seed: int,
    epochs: int,
    hidden: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> np.ndarray:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    x_train = np.vstack([item.x for item in train_sets]).astype(np.float32)
    y_train = np.concatenate([item.y for item in train_sets]).astype(np.float32)
    x_test = test_set.x.astype(np.float32)

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    model = Reranker(input_dim=x_train.shape[1], hidden=hidden, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    x_tensor = torch.from_numpy(x_train).to(device)
    y_tensor = torch.from_numpy(y_train).to(device)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        pred = model(x_tensor)
        loss = loss_fn(pred, y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(x_test).to(device)).detach().cpu().numpy()
    return pred


def evaluate_predictions(test_set: ExampleSet, pred: np.ndarray) -> dict:
    n_sent = test_set.candidate_bleurt.shape[1]
    pred_matrix = np.full((len(MODEL_CODES), n_sent), -np.inf, dtype=np.float32)
    for value, sent, candidate in zip(pred, test_set.sentence_idx, test_set.candidate_idx):
        pred_matrix[candidate, sent] = value
    selected = np.argmax(pred_matrix, axis=0)
    learned_scores = test_set.candidate_bleurt[selected, np.arange(n_sent)]
    return {
        "learned_bleurt": float(learned_scores.mean() * 100),
        "learned_top1_rate": float(
            np.mean(learned_scores == test_set.candidate_bleurt.max(axis=0))
        ),
        "learned_selected_counts": {
            MODEL_CODES[i]: int(np.sum(selected == i)) for i in range(len(MODEL_CODES))
        },
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conditions = []
    for pair_dir in sorted(data_dir.iterdir()):
        if pair_dir.is_dir() and pair_dir.name.endswith("-new"):
            for strategy in STRATEGIES:
                conditions.append(load_condition(pair_dir, strategy))

    device = choose_device(args.device)
    fold_rows = []
    for seed in args.seeds:
        for test_set in conditions:
            train_sets = [
                item
                for item in conditions
                if not (item.pair == test_set.pair and item.strategy == test_set.strategy)
            ]
            pred = train_predict(
                train_sets=train_sets,
                test_set=test_set,
                seed=seed,
                epochs=args.epochs,
                hidden=args.hidden,
                dropout=args.dropout,
                lr=args.lr,
                weight_decay=args.weight_decay,
                device=device,
            )
            metrics = evaluate_predictions(test_set, pred)
            fold_rows.append(
                {
                    "seed": seed,
                    "pair": test_set.pair,
                    "strategy": test_set.strategy,
                    "device": str(device),
                    "n_train_examples": int(sum(len(item.y) for item in train_sets)),
                    "n_test_sentences": int(test_set.candidate_bleurt.shape[1]),
                    "learned_bleurt": metrics["learned_bleurt"],
                    "meprs_avg_bleurt": test_set.meprs_avg_bleurt,
                    "best_baseline_bleurt": test_set.best_baseline_bleurt,
                    "oracle_bleurt": test_set.oracle_bleurt,
                    "learned_minus_meprs": metrics["learned_bleurt"] - test_set.meprs_avg_bleurt,
                    "learned_minus_best_baseline": metrics["learned_bleurt"]
                    - test_set.best_baseline_bleurt,
                    "learned_top1_rate": metrics["learned_top1_rate"],
                    "learned_selected_counts_json": json.dumps(
                        metrics["learned_selected_counts"], sort_keys=True
                    ),
                }
            )

    write_csv(output_dir / "learned_reranker_folds.csv", fold_rows)

    numeric = [
        "learned_bleurt",
        "meprs_avg_bleurt",
        "best_baseline_bleurt",
        "oracle_bleurt",
        "learned_minus_meprs",
        "learned_minus_best_baseline",
        "learned_top1_rate",
    ]
    summary = {
        "device": str(device),
        "conditions": len(conditions),
        "seeds": ",".join(str(seed) for seed in args.seeds),
        "epochs": args.epochs,
        "hidden": args.hidden,
    }
    for key in numeric:
        values = np.asarray([float(row[key]) for row in fold_rows], dtype=np.float64)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    write_csv(output_dir / "learned_reranker_summary.csv", [summary])

    metadata = {
        "data_dir": str(data_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "feature_dimensions": int(conditions[0].x.shape[1]) if conditions else 0,
        "model_codes": MODEL_CODES,
        "dimensions": DIMENSIONS,
        "device": str(device),
        "note": (
            "Candidate-level neural regression from LLM reviewer scores to BLEURT; "
            "selection is evaluated leave-one pair/strategy condition out."
        ),
    }
    (output_dir / "learned_reranker_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("revision_analysis") / "learned_reranker_outputs",
    )
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
