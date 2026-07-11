"""
Leakage-safe, resumable 3D-CNN cross-validation for the fixed ADNI pipeline.

For every outer fold, this script saves:
  * the best CNN state;
  * deterministic embeddings/probabilities for outer-train and outer-test;
  * OOF predictions assembled only from outer-test rows.

The fold-specific train/test embeddings allow fuse_and_report.py to fit
the fusion model without exposing an outer-test subject to CNN training.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path

import monai
import numpy as np
import pandas as pd
import torch
from monai.data import Dataset
from monai.networks.nets import resnet18
from monai.transforms import (
    Compose,
    ConcatItemsd,
    DeleteItemsd,
    EnsureChannelFirstd,
    LoadImaged,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    Resized,
    ScaleIntensityd,
    ToTensord,
)
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
IMG = (96, 96, 96)
BATCH = 4
FULL_EPOCHS = 40
FULL_FOLDS = 5
PATIENCE = 8
LR = 1e-4
SEED = 42
TASKS = ("dx3", "adcn", "conv36")
PIPELINE_VERSION = "fixed-deep-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--tasks", default="")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=HERE / "fixed_deep_manifest.csv",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU execution. Full training on CPU is not recommended.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    monai.utils.set_determinism(seed=seed)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_signature(
    manifest: Path,
    epochs: int,
    folds: int,
    smoke: bool,
) -> str:
    configuration = {
        "pipeline_version": PIPELINE_VERSION,
        "manifest_sha256": file_sha256(manifest),
        "img": IMG,
        "batch": BATCH,
        "epochs": epochs,
        "folds": folds,
        "patience": PATIENCE,
        "lr": LR,
        "seed": SEED,
        "smoke": smoke,
        "torch": torch.__version__,
        "monai": monai.__version__,
    }
    encoded = json.dumps(configuration, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_csv(frame: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_torch_save(state: dict[str, torch.Tensor], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(state, temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_manifest(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {path}. Run build_deep_manifest.py first."
        )
    manifest = pd.read_csv(path, low_memory=False)
    required = {
        "RID",
        "PTID",
        "ImageUID",
        "path_mwp1",
        "path_mwp2",
        "baseline_dx",
        "conv_36",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    if manifest.empty:
        raise ValueError("Manifest is empty")
    if manifest["RID"].duplicated().any():
        raise ValueError("Manifest contains duplicate RIDs")
    manifest["RID"] = pd.to_numeric(manifest["RID"], errors="raise").astype(int)

    missing_files: list[str] = []
    for row in manifest.itertuples():
        for column in ("path_mwp1", "path_mwp2"):
            file_path = Path(getattr(row, column))
            if not file_path.is_file():
                missing_files.append(str(file_path))
                if len(missing_files) >= 20:
                    break
        if len(missing_files) >= 20:
            break
    if missing_files:
        raise FileNotFoundError(
            "Manifest references missing CAT12 maps:\n" + "\n".join(missing_files)
        )
    return manifest


def subset(
    manifest: pd.DataFrame, task: str
) -> tuple[pd.DataFrame, str, list[str]]:
    if task == "dx3":
        classes = ["CN", "MCI", "AD"]
        data = manifest[manifest["baseline_dx"].isin(classes)]
        return data.reset_index(drop=True), "baseline_dx", classes
    if task == "adcn":
        classes = ["CN", "AD"]
        data = manifest[manifest["baseline_dx"].isin(classes)]
        return data.reset_index(drop=True), "baseline_dx", classes
    if task == "conv36":
        classes = ["sMCI", "pMCI"]
        data = manifest[
            (manifest["baseline_dx"] == "MCI")
            & manifest["conv_36"].isin(classes)
        ]
        return data.reset_index(drop=True), "conv_36", classes
    raise ValueError(f"Unknown task: {task}")


def transforms(training: bool) -> Compose:
    keys = ["mwp1", "mwp2"]
    items: list[object] = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Resized(keys=keys, spatial_size=IMG),
        ScaleIntensityd(keys=keys),
        ConcatItemsd(keys=keys, name="image", dim=0),
        DeleteItemsd(keys=keys),
    ]
    if training:
        items.extend(
            [
                RandFlipd(keys="image", prob=0.5, spatial_axis=0),
                RandAffined(
                    keys="image",
                    prob=0.3,
                    rotate_range=(0.05,) * 3,
                    scale_range=(0.05,) * 3,
                    mode="bilinear",
                ),
                RandGaussianNoised(
                    keys="image", prob=0.2, mean=0.0, std=0.02
                ),
            ]
        )
    items.append(ToTensord(keys="image"))
    return Compose(items)


def records(
    frame: pd.DataFrame, classes: list[str], y_column: str
) -> list[dict[str, object]]:
    class_to_index = {label: index for index, label in enumerate(classes)}
    output: list[dict[str, object]] = []
    for row in frame.itertuples():
        label = str(getattr(row, y_column))
        if label not in class_to_index:
            raise ValueError(f"Unexpected label {label!r} in {y_column}")
        output.append(
            {
                "mwp1": row.path_mwp1,
                "mwp2": row.path_mwp2,
                "label": class_to_index[label],
                "rid": int(row.RID),
            }
        )
    return output


def loader(dataset: Dataset, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=BATCH,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def make_model(number_of_classes: int, device: str) -> torch.nn.Module:
    return resnet18(
        spatial_dims=3,
        n_input_channels=2,
        num_classes=number_of_classes,
        shortcut_type="B",
    ).to(device)


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    class_weights: torch.Tensor,
    device: str,
) -> None:
    model.train()
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)
    for batch in data_loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device, enabled=(device == "cuda")
        ):
            logits = model(images)
            loss = loss_function(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


def infer(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    rids: list[int] = []
    captured: dict[str, torch.Tensor] = {}

    hook = model.fc.register_forward_hook(
        lambda module, inputs, output: captured.__setitem__(
            "embedding", inputs[0].detach().cpu()
        )
    )
    try:
        with torch.no_grad():
            for batch in data_loader:
                images = batch["image"].to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device, enabled=(device == "cuda")
                ):
                    logits = model(images)
                if "embedding" not in captured:
                    raise RuntimeError("CNN embedding hook did not capture features")
                probabilities.append(
                    torch.softmax(logits.float(), dim=1).cpu().numpy()
                )
                labels.append(batch["label"].numpy())
                embeddings.append(captured["embedding"].numpy())
                rids.extend(int(value) for value in batch["rid"].numpy())
    finally:
        hook.remove()

    if not labels:
        raise ValueError("Inference loader is empty")
    return (
        np.concatenate(labels),
        np.concatenate(probabilities),
        np.concatenate(embeddings),
        np.asarray(rids, dtype=int),
    )


def inference_rows(
    task: str,
    fold: int,
    split_name: str,
    classes: list[str],
    labels: np.ndarray,
    probabilities: np.ndarray,
    embeddings: np.ndarray,
    rids: np.ndarray,
    signature: str,
) -> pd.DataFrame:
    if not (
        len(labels) == len(probabilities) == len(embeddings) == len(rids)
    ):
        raise ValueError("Inference outputs have inconsistent lengths")
    rows: list[dict[str, object]] = []
    for index, rid in enumerate(rids):
        row: dict[str, object] = {
            "run_signature": signature,
            "task": task,
            "fold": fold,
            "split": split_name,
            "RID": int(rid),
            "y_true": classes[int(labels[index])],
            "y_pred": classes[int(probabilities[index].argmax())],
        }
        for class_index, class_name in enumerate(classes):
            row[f"p_{class_name}"] = float(probabilities[index, class_index])
        for embedding_index, value in enumerate(embeddings[index]):
            row[f"emb_{embedding_index:03d}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def valid_fold_artifacts(
    feature_path: Path,
    model_path: Path,
    task: str,
    fold: int,
    classes: list[str],
    expected_train_rids: set[int],
    expected_test_rids: set[int],
    signature: str,
) -> pd.DataFrame | None:
    if not feature_path.is_file() or not model_path.is_file():
        return None
    if model_path.stat().st_size < 1_000_000:
        return None
    try:
        frame = pd.read_csv(feature_path, low_memory=False)
        probability_columns = [f"p_{class_name}" for class_name in classes]
        embedding_columns = [
            column for column in frame.columns if column.startswith("emb_")
        ]
        required = {
            "run_signature",
            "task",
            "fold",
            "split",
            "RID",
            "y_true",
            "y_pred",
            *probability_columns,
        }
        if required - set(frame.columns):
            return None
        if len(embedding_columns) != 512:
            return None
        if set(frame["run_signature"].astype(str)) != {signature}:
            return None
        if set(frame["task"].astype(str)) != {task}:
            return None
        if set(pd.to_numeric(frame["fold"], errors="raise").astype(int)) != {fold}:
            return None
        if frame["RID"].duplicated().any():
            return None
        frame["RID"] = pd.to_numeric(frame["RID"], errors="raise").astype(int)
        train_rids = set(frame.loc[frame["split"] == "train", "RID"])
        test_rids = set(frame.loc[frame["split"] == "test", "RID"])
        if train_rids != expected_train_rids:
            return None
        if test_rids != expected_test_rids:
            return None
        if train_rids & test_rids:
            return None
        numeric_columns = probability_columns + embedding_columns
        values = frame[numeric_columns].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            return None
        probability_sums = frame[probability_columns].sum(axis=1).to_numpy()
        if not np.allclose(probability_sums, 1.0, atol=1e-4):
            return None
        if not set(frame["y_true"].astype(str)).issubset(set(classes)):
            return None
        return frame
    except Exception:
        return None


def auc_score(
    y_true: np.ndarray, probabilities: np.ndarray, classes: list[str]
) -> float:
    if len(classes) == 2:
        binary = (y_true == classes[1]).astype(int)
        return float(roc_auc_score(binary, probabilities[:, 1]))
    return float(
        roc_auc_score(
            label_binarize(y_true, classes=classes),
            probabilities,
            average="macro",
        )
    )


def main() -> None:
    args = parse_args()
    tasks = (
        [value.strip() for value in args.tasks.split(",") if value.strip()]
        if args.tasks
        else list(TASKS)
    )
    unknown_tasks = set(tasks) - set(TASKS)
    if unknown_tasks:
        raise ValueError(f"Unknown tasks: {sorted(unknown_tasks)}")

    epochs = 2 if args.smoke else FULL_EPOCHS
    folds = 2 if args.smoke else FULL_FOLDS
    output = (
        args.out
        if args.out is not None
        else HERE / ("fixed_results_smoke" if args.smoke else "fixed_results")
    )
    output.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and not args.allow_cpu:
        raise RuntimeError(
            "CUDA GPU is not available. Re-run RUN_0_setup.bat or use "
            "--allow-cpu explicitly."
        )
    print(
        f"device: {device} | GPU: "
        f"{torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'} | "
        f"monai {monai.__version__} | torch {torch.__version__}",
        flush=True,
    )

    manifest = validate_manifest(args.manifest.resolve())
    signature = run_signature(args.manifest.resolve(), epochs, folds, args.smoke)
    print("run signature:", signature, flush=True)

    for task_index, task in enumerate(tasks):
        data, y_column, classes = subset(manifest, task)
        y = data[y_column].astype(str).to_numpy()
        counts = pd.Series(y).value_counts()
        if set(counts.index) != set(classes):
            raise ValueError(
                f"{task}: missing classes; expected {classes}, found {counts.to_dict()}"
            )
        if int(counts.min()) < folds:
            raise ValueError(
                f"{task}: smallest class has {counts.min()} rows, below {folds} folds"
            )

        splitter = StratifiedKFold(
            n_splits=folds, shuffle=True, random_state=SEED
        )
        oof_parts: list[pd.DataFrame] = []
        for fold, (train_indices, test_indices) in enumerate(
            splitter.split(data, y)
        ):
            outer_train = data.iloc[train_indices].reset_index(drop=True)
            outer_test = data.iloc[test_indices].reset_index(drop=True)
            expected_train_rids = set(outer_train["RID"].astype(int))
            expected_test_rids = set(outer_test["RID"].astype(int))
            if expected_train_rids & expected_test_rids:
                raise RuntimeError(f"{task} fold{fold}: train/test RID leakage")

            feature_path = output / f"deep_features_{task}_fold{fold}.csv"
            model_path = output / f"model_{task}_fold{fold}.pt"
            existing = None
            if not args.smoke:
                existing = valid_fold_artifacts(
                    feature_path,
                    model_path,
                    task,
                    fold,
                    classes,
                    expected_train_rids,
                    expected_test_rids,
                    signature,
                )
            if existing is not None:
                print(f"skip (complete): {task} fold{fold}", flush=True)
                oof_parts.append(existing[existing["split"] == "test"].copy())
                continue

            fold_seed = SEED + task_index * 100 + fold
            set_seed(fold_seed)
            outer_train_y = outer_train[y_column].astype(str).to_numpy()
            inner_train_indices, validation_indices = train_test_split(
                np.arange(len(outer_train)),
                test_size=0.15,
                stratify=outer_train_y,
                random_state=fold_seed,
            )
            inner_train = outer_train.iloc[inner_train_indices]
            validation = outer_train.iloc[validation_indices]

            train_dataset = Dataset(
                records(inner_train, classes, y_column), transforms(True)
            )
            validation_dataset = Dataset(
                records(validation, classes, y_column), transforms(False)
            )
            model = make_model(len(classes), device)

            training_labels = [
                item["label"] for item in records(inner_train, classes, y_column)
            ]
            class_counts = np.bincount(
                training_labels, minlength=len(classes)
            )
            if np.any(class_counts == 0):
                raise ValueError(
                    f"{task} fold{fold}: inner training split lacks a class"
                )
            class_weights = torch.tensor(
                class_counts.sum() / (len(classes) * class_counts),
                dtype=torch.float32,
                device=device,
            )
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=LR, weight_decay=1e-4
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs
            )
            scaler = torch.amp.GradScaler(
                "cuda", enabled=(device == "cuda")
            )

            best_score = -1.0
            best_state: dict[str, torch.Tensor] | None = None
            wait = 0
            for epoch in range(epochs):
                train_one_epoch(
                    model,
                    loader(train_dataset, True),
                    optimizer,
                    scaler,
                    class_weights,
                    device,
                )
                scheduler.step()
                val_labels, val_probabilities, _, _ = infer(
                    model, loader(validation_dataset, False), device
                )
                val_true = np.asarray(classes)[val_labels]
                val_predicted = np.asarray(classes)[
                    val_probabilities.argmax(axis=1)
                ]
                score = balanced_accuracy_score(val_true, val_predicted)
                improved = score > best_score
                if improved:
                    best_score = float(score)
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                    wait = 0
                else:
                    wait += 1
                marker = " *" if improved else ""
                print(
                    f"  {task} fold{fold} ep{epoch:02d} "
                    f"val_bAcc={score:.3f}{marker}",
                    flush=True,
                )
                if wait >= PATIENCE:
                    break

            if best_state is None:
                raise RuntimeError(f"{task} fold{fold}: no best model state")
            model.load_state_dict(best_state)

            deterministic_transform = transforms(False)
            train_features = infer(
                model,
                loader(
                    Dataset(
                        records(outer_train, classes, y_column),
                        deterministic_transform,
                    ),
                    False,
                ),
                device,
            )
            test_features = infer(
                model,
                loader(
                    Dataset(
                        records(outer_test, classes, y_column),
                        transforms(False),
                    ),
                    False,
                ),
                device,
            )
            train_frame = inference_rows(
                task,
                fold,
                "train",
                classes,
                *train_features,
                signature,
            )
            test_frame = inference_rows(
                task,
                fold,
                "test",
                classes,
                *test_features,
                signature,
            )
            fold_frame = pd.concat(
                [train_frame, test_frame], ignore_index=True
            )

            atomic_torch_save(best_state, model_path)
            atomic_csv(fold_frame, feature_path)
            validated = valid_fold_artifacts(
                feature_path,
                model_path,
                task,
                fold,
                classes,
                expected_train_rids,
                expected_test_rids,
                signature,
            )
            if validated is None:
                raise RuntimeError(
                    f"{task} fold{fold}: saved artifacts failed validation"
                )
            oof_parts.append(validated[validated["split"] == "test"].copy())

            del model, optimizer, scheduler, scaler
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

        oof = pd.concat(oof_parts, ignore_index=True)
        if len(oof) != len(data):
            raise RuntimeError(
                f"{task}: OOF has {len(oof)} rows; expected {len(data)}"
            )
        if oof["RID"].duplicated().any():
            raise RuntimeError(f"{task}: duplicate RIDs in OOF predictions")
        if set(oof["RID"].astype(int)) != set(data["RID"].astype(int)):
            raise RuntimeError(f"{task}: OOF RID set does not match task cohort")
        oof = oof.sort_values("RID").reset_index(drop=True)
        oof_path = output / f"deep_oof_{task}.csv"
        atomic_csv(oof, oof_path)

        probability_columns = [f"p_{name}" for name in classes]
        y_true = oof["y_true"].astype(str).to_numpy()
        probabilities = oof[probability_columns].to_numpy(dtype=float)
        bacc = balanced_accuracy_score(
            y_true, oof["y_pred"].astype(str).to_numpy()
        )
        auc = auc_score(y_true, probabilities, classes)
        print(
            f"[OOF] {task} n={len(oof)} bAcc={bacc:.3f} "
            f"AUC={auc:.3f} -> {oof_path}",
            flush=True,
        )

    print(
        "Training done. Run fuse_and_report.py for leakage-safe fusion.",
        flush=True,
    )


if __name__ == "__main__":
    main()
