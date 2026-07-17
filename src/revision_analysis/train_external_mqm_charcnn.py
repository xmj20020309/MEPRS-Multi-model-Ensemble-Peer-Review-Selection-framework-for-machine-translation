"""Train a self-contained char-CNN MQM predictor on public WMT data.

This experiment is a GPU-friendly fallback when pretrained Transformer weights
cannot be downloaded. It uses public TEaR/WMT MQM rows with source sentences,
references, hypotheses, and human MQM scores. Evaluation is leave-one-language
pair out, so each fold tests transfer to an unseen public WMT direction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau, pearsonr, spearmanr
from torch import nn
from torch.utils.data import DataLoader, Dataset


REQUIRED_COLUMNS = ["Target_Index", "system", "src", "ref", "hyps", "mqm_score"]
TEXT_COLUMNS = ["src", "ref", "hyps"]
PAD_ID = 0
UNK_ID = 1


@dataclass
class EncodedSplit:
    x_train: np.ndarray
    f_train: np.ndarray
    y_train: np.ndarray
    x_valid: np.ndarray
    f_valid: np.ndarray
    y_valid: np.ndarray
    x_test: np.ndarray
    f_test: np.ndarray
    y_test: np.ndarray
    test_frame: pd.DataFrame
    label_mean: float
    label_std: float
    vocab: dict[str, int]
    feature_mean: np.ndarray
    feature_std: np.ndarray


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
        tensor = torch.tensor([1.0], device="cuda")
        _ = tensor * 2.0
        torch.cuda.synchronize()
        return True
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
    if data_dir.is_file():
        paths = [data_dir]
    else:
        paths = sorted(data_dir.glob("*_final_file_filtered.csv"))
        if not paths:
            paths = sorted(data_dir.glob("**/*_final_file_filtered.csv"))
        if not paths:
            paths = sorted(data_dir.glob("*segment_scores.csv"))
        if not paths:
            paths = sorted(data_dir.glob("**/*segment_scores.csv"))
    if not paths:
        raise FileNotFoundError(f"No public MQM CSV files found under {data_dir}")

    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        if "language_pair" not in frame.columns:
            frame["language_pair"] = language_pair_from_path(path)
        if "source_file" not in frame.columns:
            frame["source_file"] = path.name
        frame = frame[[*REQUIRED_COLUMNS, "language_pair", "source_file"]].copy()
        frame["mqm_score"] = pd.to_numeric(frame["mqm_score"], errors="coerce")
        frame[TEXT_COLUMNS] = frame[TEXT_COLUMNS].fillna("")
        frame = frame.dropna(subset=["src", "hyps", "mqm_score"])
        frame[TEXT_COLUMNS] = frame[TEXT_COLUMNS].astype(str)
        if smoke_rows_per_pair:
            frame = frame.groupby("language_pair", group_keys=False).head(smoke_rows_per_pair)
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    data["row_id"] = np.arange(len(data), dtype=np.int64)
    return data


def make_text(row: pd.Series, input_mode: str) -> str:
    if input_mode == "hyp_only":
        return f"<hyp> {row['hyps']}"
    if input_mode == "ref_hyp":
        return f"<ref> {row['ref']} <hyp> {row['hyps']}"
    return f"<src> {row['src']} <ref> {row['ref']} <hyp> {row['hyps']}"


def build_vocab(texts: list[str], min_count: int, max_vocab: int) -> dict[str, int]:
    counts = Counter()
    for text in texts:
        counts.update(text)
    vocab = {"<pad>": PAD_ID, "<unk>": UNK_ID}
    for char, count in counts.most_common(max_vocab - len(vocab)):
        if count >= min_count and char not in vocab:
            vocab[char] = len(vocab)
    return vocab


def encode_texts(texts: list[str], vocab: dict[str, int], max_chars: int) -> np.ndarray:
    encoded = np.full((len(texts), max_chars), PAD_ID, dtype=np.int64)
    for row_i, text in enumerate(texts):
        ids = [vocab.get(char, UNK_ID) for char in text[:max_chars]]
        if ids:
            encoded[row_i, : len(ids)] = ids
    return encoded


def char_f1(a: str, b: str) -> float:
    a_counts = Counter(a)
    b_counts = Counter(b)
    if not a_counts or not b_counts:
        return 0.0
    overlap = sum(min(a_counts[key], b_counts[key]) for key in a_counts.keys() & b_counts.keys())
    precision = overlap / sum(b_counts.values())
    recall = overlap / sum(a_counts.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def raw_features(frame: pd.DataFrame, input_mode: str) -> np.ndarray:
    rows = []
    for _, row in frame.iterrows():
        src = str(row["src"])
        ref = str(row["ref"])
        hyp = str(row["hyps"])
        src_len = max(1, len(src))
        ref_len = max(1, len(ref))
        hyp_len = max(1, len(hyp))
        if input_mode == "hyp_only":
            rows.append(
                [
                    math.log1p(hyp_len),
                    len(set(hyp)) / hyp_len,
                    sum(char.isdigit() for char in hyp) / hyp_len,
                    sum(char.isalpha() for char in hyp) / hyp_len,
                    sum(char.isspace() for char in hyp) / hyp_len,
                ]
            )
        elif input_mode == "ref_hyp":
            rows.append(
                [
                    math.log1p(ref_len),
                    math.log1p(hyp_len),
                    hyp_len / ref_len,
                    char_f1(ref, hyp),
                    len(set(hyp)) / hyp_len,
                ]
            )
        else:
            rows.append(
                [
                    math.log1p(src_len),
                    math.log1p(ref_len),
                    math.log1p(hyp_len),
                    hyp_len / ref_len,
                    hyp_len / src_len,
                    char_f1(ref, hyp),
                    char_f1(src, hyp),
                ]
            )
    return np.asarray(rows, dtype=np.float32)


def split_leave_one_pair(data: pd.DataFrame, heldout_pair: str, seed: int, valid_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test = data[data["language_pair"] == heldout_pair].copy()
    train_valid = data[data["language_pair"] != heldout_pair].copy()
    rng = np.random.default_rng(seed)
    valid_indices = []
    for _, pair_frame in train_valid.groupby("language_pair"):
        targets = np.asarray(sorted(pair_frame["Target_Index"].unique()))
        rng.shuffle(targets)
        valid_count = max(1, int(round(len(targets) * valid_fraction)))
        valid_targets = set(targets[:valid_count])
        valid_indices.extend(pair_frame[pair_frame["Target_Index"].isin(valid_targets)].index.tolist())
    valid = train_valid.loc[valid_indices].copy()
    train = train_valid.drop(index=valid_indices).copy()
    return train, valid, test


def encode_split(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    min_char_count: int,
    max_vocab: int,
    max_chars: int,
    input_mode: str,
) -> EncodedSplit:
    train_texts = [make_text(row, input_mode) for _, row in train.iterrows()]
    valid_texts = [make_text(row, input_mode) for _, row in valid.iterrows()]
    test_texts = [make_text(row, input_mode) for _, row in test.iterrows()]
    vocab = build_vocab(train_texts, min_char_count, max_vocab)

    f_train = raw_features(train, input_mode)
    f_valid = raw_features(valid, input_mode)
    f_test = raw_features(test, input_mode)
    feature_mean = f_train.mean(axis=0, keepdims=True)
    feature_std = f_train.std(axis=0, keepdims=True)
    feature_std[feature_std < 1e-6] = 1.0

    y_train_raw = train["mqm_score"].to_numpy(dtype=np.float32)
    label_mean = float(y_train_raw.mean())
    label_std = float(y_train_raw.std())
    if not math.isfinite(label_std) or label_std < 1e-6:
        label_std = 1.0

    return EncodedSplit(
        x_train=encode_texts(train_texts, vocab, max_chars),
        f_train=((f_train - feature_mean) / feature_std).astype(np.float32),
        y_train=((y_train_raw - label_mean) / label_std).astype(np.float32),
        x_valid=encode_texts(valid_texts, vocab, max_chars),
        f_valid=((f_valid - feature_mean) / feature_std).astype(np.float32),
        y_valid=((valid["mqm_score"].to_numpy(dtype=np.float32) - label_mean) / label_std).astype(np.float32),
        x_test=encode_texts(test_texts, vocab, max_chars),
        f_test=((f_test - feature_mean) / feature_std).astype(np.float32),
        y_test=((test["mqm_score"].to_numpy(dtype=np.float32) - label_mean) / label_std).astype(np.float32),
        test_frame=test.reset_index(drop=True).copy(),
        label_mean=label_mean,
        label_std=label_std,
        vocab=vocab,
        feature_mean=feature_mean.astype(np.float32),
        feature_std=feature_std.astype(np.float32),
    )


class ArrayDataset(Dataset):
    def __init__(self, ids: np.ndarray, features: np.ndarray, labels: np.ndarray) -> None:
        self.ids = ids
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.ids[index]),
            torch.from_numpy(self.features[index]),
            torch.tensor(self.labels[index], dtype=torch.float32),
        )


class CharCNNRegressor(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        feature_dim: int,
        char_dim: int,
        channels: int,
        kernels: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, char_dim, padding_idx=PAD_ID)
        self.convs = nn.ModuleList(
            nn.Conv1d(char_dim, channels, kernel_size=kernel, padding=kernel // 2)
            for kernel in kernels
        )
        pooled_dim = channels * len(kernels)
        self.head = nn.Sequential(
            nn.Linear(pooled_dim + feature_dim, pooled_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(pooled_dim, max(16, pooled_dim // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(16, pooled_dim // 2), 1),
        )

    def forward(self, ids: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        x = self.embedding(ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            value = torch.relu(conv(x))
            pooled.append(torch.amax(value, dim=-1))
        return self.head(torch.cat([*pooled, features], dim=-1)).squeeze(-1)


def train_predict(
    encoded: EncodedSplit,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    checkpoint_path: Path | None = None,
) -> tuple[np.ndarray, dict]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_dataset = ArrayDataset(encoded.x_train, encoded.f_train, encoded.y_train)
    valid_dataset = ArrayDataset(encoded.x_valid, encoded.f_valid, encoded.y_valid)
    test_dataset = ArrayDataset(encoded.x_test, encoded.f_test, encoded.y_test)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size * 2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size * 2, shuffle=False)

    model = CharCNNRegressor(
        vocab_size=int(max(encoded.x_train.max(), encoded.x_valid.max(), encoded.x_test.max()) + 1),
        feature_dim=encoded.f_train.shape[1],
        char_dim=args.char_dim,
        channels=args.channels,
        kernels=args.kernels,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    best_state = None
    best_valid = float("inf")
    bad_epochs = 0
    history = []
    best_epoch = 0
    completed_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for ids, features, labels in train_loader:
            ids = ids.to(device)
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(ids, features), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        valid_loss = evaluate_loss(model, valid_loader, device, loss_fn)
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        history.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss})
        completed_epochs = epoch
        if valid_loss < best_valid:
            best_valid = valid_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
        if args.patience and bad_epochs >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": best_state,
                    "vocab": encoded.vocab,
                    "feature_mean": encoded.feature_mean,
                    "feature_std": encoded.feature_std,
                    "label_mean": encoded.label_mean,
                    "label_std": encoded.label_std,
                    "model_config": {
                        "vocab_size": int(max(encoded.x_train.max(), encoded.x_valid.max(), encoded.x_test.max()) + 1),
                        "feature_dim": int(encoded.f_train.shape[1]),
                        "char_dim": args.char_dim,
                        "channels": args.channels,
                        "kernels": args.kernels,
                        "dropout": args.dropout,
                        "max_chars": args.max_chars,
                        "input_mode": args.input_mode,
                    },
                    "training": {
                        "seed": seed,
                        "best_epoch": best_epoch,
                        "completed_epochs": completed_epochs,
                        "best_valid_loss": best_valid,
                        "history": history,
                    },
                },
                checkpoint_path,
            )

    pred_scaled = predict(model, test_loader, device)
    train_info = {
        "best_epoch": best_epoch,
        "completed_epochs": completed_epochs,
        "best_valid_loss": best_valid,
    }
    return pred_scaled * encoded.label_std + encoded.label_mean, train_info


def evaluate_loss(model: nn.Module, loader: DataLoader, device: torch.device, loss_fn: nn.Module) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for ids, features, labels in loader:
            ids = ids.to(device)
            features = features.to(device)
            labels = labels.to(device)
            losses.append(float(loss_fn(model(ids, features), labels).detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    values = []
    with torch.no_grad():
        for ids, features, _ in loader:
            ids = ids.to(device)
            features = features.to(device)
            values.extend(model(ids, features).detach().cpu().numpy().tolist())
    return np.asarray(values, dtype=np.float32)


def safe_corr(function, gold: np.ndarray, pred: np.ndarray) -> float:
    if len(gold) < 2 or np.std(gold) < 1e-8 or np.std(pred) < 1e-8:
        return float("nan")
    value = function(gold, pred)[0]
    return float(value) if value is not None else float("nan")


def pairwise_accuracy(frame: pd.DataFrame, pred_column: str) -> float:
    correct = 0
    total = 0
    for _, group in frame.groupby("Target_Index"):
        gold = group["mqm_score"].to_numpy(dtype=float)
        pred = group[pred_column].to_numpy(dtype=float)
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


def top1_accuracy(frame: pd.DataFrame, pred_column: str) -> float:
    hits = []
    for _, group in frame.groupby("Target_Index"):
        gold_best = set(group[group["mqm_score"] == group["mqm_score"].max()]["system"].tolist())
        pred_system = str(group.loc[group[pred_column].idxmax(), "system"])
        hits.append(pred_system in gold_best)
    return float(np.mean(hits)) if hits else float("nan")


def add_overlap_baseline(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["overlap_baseline"] = [
        char_f1(str(row["ref"]), str(row["hyps"])) for _, row in frame.iterrows()
    ]
    return frame


def summarize_fold(frame: pd.DataFrame, seed: int, heldout_pair: str) -> dict:
    gold = frame["mqm_score"].to_numpy(dtype=float)
    pred = frame["pred_mqm_score"].to_numpy(dtype=float)
    base = frame["overlap_baseline"].to_numpy(dtype=float)
    system = frame.groupby("system", as_index=False)[
        ["mqm_score", "pred_mqm_score", "overlap_baseline"]
    ].mean()
    system_gold = system["mqm_score"].to_numpy(dtype=float)
    system_pred = system["pred_mqm_score"].to_numpy(dtype=float)
    system_base = system["overlap_baseline"].to_numpy(dtype=float)
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
        "pairwise_accuracy": pairwise_accuracy(frame, "pred_mqm_score"),
        "top1_accuracy": top1_accuracy(frame, "pred_mqm_score"),
        "baseline_pearson": safe_corr(pearsonr, gold, base),
        "baseline_spearman": safe_corr(spearmanr, gold, base),
        "baseline_system_spearman": safe_corr(spearmanr, system_gold, system_base),
        "baseline_pairwise_accuracy": pairwise_accuracy(frame, "overlap_baseline"),
        "baseline_top1_accuracy": top1_accuracy(frame, "overlap_baseline"),
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
    set_seed(args.seeds[0])
    data = load_public_mqm(Path(args.data_dir), smoke_rows_per_pair=args.smoke_rows_per_pair)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    fold_rows = []
    prediction_frames = []
    for seed in args.seeds:
        set_seed(seed)
        for heldout_pair in sorted(data["language_pair"].unique()):
            train, valid, test = split_leave_one_pair(data, heldout_pair, seed, args.valid_fraction)
            encoded = encode_split(
                train=train,
                valid=valid,
                test=test,
                min_char_count=args.min_char_count,
                max_vocab=args.max_vocab,
                max_chars=args.max_chars,
                input_mode=args.input_mode,
            )
            checkpoint_path = None
            if args.save_checkpoints:
                checkpoint_path = (
                    output_dir
                    / "checkpoints"
                    / f"charcnn_seed{seed}_{heldout_pair.replace('-', '')}.pt"
                )
            predictions, train_info = train_predict(encoded, seed, args, device, checkpoint_path)
            pred_frame = encoded.test_frame.copy()
            pred_frame["seed"] = seed
            pred_frame["pred_mqm_score"] = predictions
            pred_frame = add_overlap_baseline(pred_frame)
            prediction_frames.append(pred_frame)
            fold_row = summarize_fold(pred_frame, seed, heldout_pair)
            fold_row.update(
                {
                    "best_epoch": train_info["best_epoch"],
                    "completed_epochs": train_info["completed_epochs"],
                    "best_valid_loss": train_info["best_valid_loss"],
                    "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else "",
                }
            )
            fold_rows.append(fold_row)
            write_csv(output_dir / "external_mqm_charcnn_folds.partial.csv", fold_rows)

    metric_keys = [
        "pearson",
        "spearman",
        "kendall",
        "system_pearson",
        "system_spearman",
        "pairwise_accuracy",
        "top1_accuracy",
        "baseline_pearson",
        "baseline_spearman",
        "baseline_system_spearman",
        "baseline_pairwise_accuracy",
        "baseline_top1_accuracy",
    ]
    summary = {
        "device": str(device),
        "language_pairs": ";".join(sorted(data["language_pair"].unique())),
        "rows": int(len(data)),
        "segments": int(data[["language_pair", "Target_Index"]].drop_duplicates().shape[0]),
        "seeds": ",".join(str(seed) for seed in args.seeds),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_chars": args.max_chars,
        "char_dim": args.char_dim,
        "channels": args.channels,
        "input_mode": args.input_mode,
    }
    for key in metric_keys:
        values = np.asarray([row[key] for row in fold_rows], dtype=float)
        summary[f"{key}_mean"] = float(np.nanmean(values))
        summary[f"{key}_sd"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
    summary["spearman_minus_baseline_mean"] = (
        summary["spearman_mean"] - summary["baseline_spearman_mean"]
    )
    summary["pairwise_minus_baseline_mean"] = (
        summary["pairwise_accuracy_mean"] - summary["baseline_pairwise_accuracy_mean"]
    )

    write_csv(output_dir / "external_mqm_charcnn_folds.csv", fold_rows)
    write_csv(output_dir / "external_mqm_charcnn_summary.csv", [summary])
    if prediction_frames and not args.skip_predictions:
        pd.concat(prediction_frames, ignore_index=True).to_csv(
            output_dir / "external_mqm_charcnn_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
    metadata = {
        "data_dir": args.data_dir,
        "output_dir": str(output_dir),
        "device": str(device),
        "rows_by_pair": data.groupby("language_pair").size().to_dict(),
        "segments_by_pair": data.groupby("language_pair")["Target_Index"].nunique().to_dict(),
        "arguments": vars(args),
    }
    (output_dir / "external_mqm_charcnn_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="public_data")
    parser.add_argument("--output_dir", default="revision_analysis/external_mqm_charcnn_outputs")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_chars", type=int, default=768)
    parser.add_argument(
        "--input_mode",
        choices=["full", "ref_hyp", "hyp_only"],
        default="full",
        help="Text and feature fields exposed to the model.",
    )
    parser.add_argument("--char_dim", type=int, default=96)
    parser.add_argument("--channels", type=int, default=96)
    parser.add_argument("--kernels", type=int, nargs="+", default=[3, 5, 7, 11])
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--valid_fraction", type=float, default=0.1)
    parser.add_argument("--min_char_count", type=int, default=2)
    parser.add_argument("--max_vocab", type=int, default=6000)
    parser.add_argument("--smoke_rows_per_pair", type=int, default=None)
    parser.add_argument(
        "--skip_predictions",
        action="store_true",
        help="Write fold summaries only; useful for large Google MQM runs.",
    )
    parser.add_argument(
        "--save_checkpoints",
        action="store_true",
        help="Save the best model state, vocab, and normalization values for each fold.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
