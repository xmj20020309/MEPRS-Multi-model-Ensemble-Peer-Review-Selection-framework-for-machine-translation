"""Train a multilingual MQM quality predictor on public WMT data.

This revision experiment uses public TEaR/WMT MQM files with source sentences,
references, system hypotheses, and human MQM scores. A multilingual Transformer
is fine-tuned as a regression model and evaluated with leave-one-language-pair
out validation. The experiment is intended to use a GPU when available and does
not call any LLM APIs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau, pearsonr, spearmanr
from torch.utils.data import DataLoader, Dataset


TEXT_COLUMNS = ["src", "ref", "hyps"]
REQUIRED_COLUMNS = ["Target_Index", "system", "src", "ref", "hyps", "mqm_score"]


@dataclass
class SplitData:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_cuda_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        value = torch.tensor([1.0], device="cuda") * 2
        torch.cuda.synchronize()
        return float(value.cpu()[0]) == 2.0
    except Exception:
        return False


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda" if safe_cuda_available() else "cpu")
    return torch.device("cuda" if safe_cuda_available() else "cpu")


def language_pair_from_path(path: Path) -> str:
    match = re.search(r"__([a-z]{2}-[a-z]{2})__", path.name)
    if match:
        return match.group(1)
    match = re.search(r"([a-z]{2}-[a-z]{2})_final_file_filtered", path.name)
    if match:
        return match.group(1)
    raise ValueError(f"Cannot infer language pair from {path}")


def load_public_mqm(data_dir: Path, smoke_rows_per_pair: int | None = None) -> pd.DataFrame:
    files = sorted(data_dir.glob("*_final_file_filtered.csv"))
    if not files:
        files = sorted(data_dir.glob("**/*_final_file_filtered.csv"))
    if not files:
        raise FileNotFoundError(f"No *_final_file_filtered.csv files found under {data_dir}")

    frames = []
    for path in files:
        frame = pd.read_csv(path)
        missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        frame = frame[REQUIRED_COLUMNS].copy()
        frame["language_pair"] = language_pair_from_path(path)
        frame["source_file"] = path.name
        frame["mqm_score"] = pd.to_numeric(frame["mqm_score"], errors="coerce")
        frame = frame.dropna(subset=["mqm_score", *TEXT_COLUMNS])
        if smoke_rows_per_pair:
            frame = frame.groupby("language_pair", group_keys=False).head(smoke_rows_per_pair)
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    data["row_id"] = np.arange(len(data), dtype=np.int64)
    return data


def make_input(row: pd.Series) -> str:
    return f"Source: {row['src']}\nReference: {row['ref']}\nHypothesis: {row['hyps']}"


class MQMDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer, max_length: int, label_mean: float, label_std: float) -> None:
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_mean = label_mean
        self.label_std = label_std

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.frame.iloc[index]
        encoded = self.tokenizer(
            make_input(row),
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        label = (float(row["mqm_score"]) - self.label_mean) / self.label_std
        encoded["labels"] = float(label)
        return encoded


def make_collator(tokenizer):
    def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
        labels = torch.tensor([item.pop("labels") for item in batch], dtype=torch.float32)
        encoded = tokenizer.pad(batch, padding=True, return_tensors="pt")
        encoded["labels"] = labels
        return encoded

    return collate


def split_leave_one_pair(data: pd.DataFrame, heldout_pair: str, seed: int, valid_fraction: float) -> SplitData:
    test = data[data["language_pair"] == heldout_pair].copy()
    train_valid = data[data["language_pair"] != heldout_pair].copy()
    rng = np.random.default_rng(seed)

    valid_indices = []
    for pair, pair_frame in train_valid.groupby("language_pair"):
        targets = np.asarray(sorted(pair_frame["Target_Index"].unique()))
        rng.shuffle(targets)
        n_valid = max(1, int(round(len(targets) * valid_fraction)))
        valid_targets = set(targets[:n_valid])
        valid_indices.extend(pair_frame[pair_frame["Target_Index"].isin(valid_targets)].index.tolist())

    valid = train_valid.loc[valid_indices].copy()
    train = train_valid.drop(index=valid_indices).copy()
    return SplitData(train=train, valid=valid, test=test)


def train_one_fold(
    split: SplitData,
    tokenizer,
    model_name: str,
    cache_dir: str | None,
    seed: int,
    epochs: int,
    batch_size: int,
    max_length: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> np.ndarray:
    from transformers import AutoModelForSequenceClassification

    label_mean = float(split.train["mqm_score"].mean())
    label_std = float(split.train["mqm_score"].std(ddof=0))
    if not math.isfinite(label_std) or label_std < 1e-6:
        label_std = 1.0

    train_dataset = MQMDataset(split.train, tokenizer, max_length, label_mean, label_std)
    valid_dataset = MQMDataset(split.valid, tokenizer, max_length, label_mean, label_std)
    test_dataset = MQMDataset(split.test, tokenizer, max_length, label_mean, label_std)
    collator = make_collator(tokenizer)

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collator,
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size * 2, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False, collate_fn=collator)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        problem_type="regression",
        cache_dir=cache_dir,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()

    best_state = None
    best_valid = float("inf")
    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            pred = model(**batch).logits.squeeze(-1)
            loss = loss_fn(pred, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        valid_loss = evaluate_loss(model, valid_loader, device, loss_fn)
        if valid_loss < best_valid:
            best_valid = valid_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    pred_scaled = predict(model, test_loader, device)
    return pred_scaled * label_std + label_mean


def evaluate_loss(model, loader: DataLoader, device: torch.device, loss_fn) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            pred = model(**batch).logits.squeeze(-1)
            losses.append(float(loss_fn(pred, labels).detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def predict(model, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    values = []
    with torch.no_grad():
        for batch in loader:
            _ = batch.pop("labels")
            batch = {key: value.to(device) for key, value in batch.items()}
            values.extend(model(**batch).logits.squeeze(-1).detach().cpu().numpy().tolist())
    return np.asarray(values, dtype=np.float32)


def safe_corr(function, gold: np.ndarray, pred: np.ndarray) -> float:
    if len(gold) < 2 or np.std(gold) < 1e-8 or np.std(pred) < 1e-8:
        return float("nan")
    value = function(gold, pred)[0]
    return float(value) if value is not None else float("nan")


def pairwise_accuracy(frame: pd.DataFrame) -> float:
    correct = 0
    total = 0
    for _, group in frame.groupby("Target_Index"):
        gold = group["mqm_score"].to_numpy(dtype=float)
        pred = group["pred_mqm_score"].to_numpy(dtype=float)
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                gold_delta = gold[i] - gold[j]
                pred_delta = pred[i] - pred[j]
                if abs(gold_delta) < 1e-8:
                    continue
                total += 1
                if gold_delta * pred_delta > 0:
                    correct += 1
    return float(correct / total) if total else float("nan")


def top1_accuracy(frame: pd.DataFrame) -> float:
    hits = []
    for _, group in frame.groupby("Target_Index"):
        gold_best = set(group[group["mqm_score"] == group["mqm_score"].max()]["system"].tolist())
        pred_system = str(group.loc[group["pred_mqm_score"].idxmax(), "system"])
        hits.append(pred_system in gold_best)
    return float(np.mean(hits)) if hits else float("nan")


def summarize_fold(frame: pd.DataFrame, seed: int, heldout_pair: str) -> dict:
    gold = frame["mqm_score"].to_numpy(dtype=float)
    pred = frame["pred_mqm_score"].to_numpy(dtype=float)
    system = frame.groupby("system", as_index=False)[["mqm_score", "pred_mqm_score"]].mean()
    system_gold = system["mqm_score"].to_numpy(dtype=float)
    system_pred = system["pred_mqm_score"].to_numpy(dtype=float)
    return {
        "seed": seed,
        "heldout_pair": heldout_pair,
        "test_rows": int(len(frame)),
        "test_segments": int(frame["Target_Index"].nunique()),
        "test_systems": int(frame["system"].nunique()),
        "pearson": safe_corr(pearsonr, gold, pred),
        "spearman": safe_corr(spearmanr, gold, pred),
        "kendall": safe_corr(kendalltau, gold, pred),
        "system_pearson": safe_corr(pearsonr, system_gold, system_pred),
        "system_spearman": safe_corr(spearmanr, system_gold, system_pred),
        "pairwise_accuracy": pairwise_accuracy(frame),
        "top1_accuracy": top1_accuracy(frame),
        "gold_mean": float(np.mean(gold)),
        "pred_mean": float(np.mean(pred)),
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
    from transformers import AutoTokenizer

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_public_mqm(data_dir, smoke_rows_per_pair=args.smoke_rows_per_pair)
    device = choose_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir, use_fast=True)

    fold_rows = []
    prediction_frames = []
    for seed in args.seeds:
        set_seed(seed)
        for heldout_pair in sorted(data["language_pair"].unique()):
            split = split_leave_one_pair(data, heldout_pair, seed, args.valid_fraction)
            predictions = train_one_fold(
                split=split,
                tokenizer=tokenizer,
                model_name=args.model_name,
                cache_dir=args.cache_dir,
                seed=seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                max_length=args.max_length,
                lr=args.lr,
                weight_decay=args.weight_decay,
                device=device,
            )
            pred_frame = split.test.copy().reset_index(drop=True)
            pred_frame["seed"] = seed
            pred_frame["pred_mqm_score"] = predictions
            prediction_frames.append(pred_frame)
            fold_rows.append(summarize_fold(pred_frame, seed, heldout_pair))

    summary_keys = [
        "pearson",
        "spearman",
        "kendall",
        "system_pearson",
        "system_spearman",
        "pairwise_accuracy",
        "top1_accuracy",
    ]
    summary = {
        "device": str(device),
        "model_name": args.model_name,
        "language_pairs": ";".join(sorted(data["language_pair"].unique())),
        "rows": int(len(data)),
        "segments": int(data[["language_pair", "Target_Index"]].drop_duplicates().shape[0]),
        "seeds": ",".join(str(seed) for seed in args.seeds),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
    }
    for key in summary_keys:
        values = np.asarray([row[key] for row in fold_rows], dtype=float)
        summary[f"{key}_mean"] = float(np.nanmean(values))
        summary[f"{key}_sd"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0

    write_csv(output_dir / "external_mqm_predictor_folds.csv", fold_rows)
    write_csv(output_dir / "external_mqm_predictor_summary.csv", [summary])
    if prediction_frames:
        predictions = pd.concat(prediction_frames, ignore_index=True)
        predictions.to_csv(output_dir / "external_mqm_predictor_predictions.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "device": str(device),
        "model_name": args.model_name,
        "cache_dir": args.cache_dir,
        "rows_by_pair": data.groupby("language_pair").size().to_dict(),
        "segments_by_pair": data.groupby("language_pair")["Target_Index"].nunique().to_dict(),
        "columns": list(data.columns),
    }
    (output_dir / "external_mqm_predictor_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="public_data", help="Directory containing public MQM CSV files.")
    parser.add_argument("--output_dir", default="revision_analysis/external_mqm_predictor_outputs")
    parser.add_argument("--model_name", default="distilbert-base-multilingual-cased")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--valid_fraction", type=float, default=0.1)
    parser.add_argument("--smoke_rows_per_pair", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
