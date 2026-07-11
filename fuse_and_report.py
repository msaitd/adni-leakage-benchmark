"""
Leakage-safe fold-aligned fusion for train_cnn_cv.py outputs.

For each task and outer fold, all fusion models are fitted exclusively on that
fold's outer-train subjects and evaluated on its untouched outer-test subjects.
Predictions are then concatenated once per subject to form true OOF metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize


HERE = Path(__file__).resolve().parent
TASKS = ("dx3", "adcn", "conv36")
FOLDS = 5
SEED = 42
FEATURESETS = ("clinical", "freesurfer", "deep_cnn", "clinical+deep", "all")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results", type=Path, default=HERE / "fixed_results"
    )
    parser.add_argument(
        "--data", type=Path, default=HERE.parent / "data"
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=HERE / "fixed_deep_final",
        help="Prefix for _summary.csv, _oof.csv, and _comparison.png",
    )
    return parser.parse_args()


def classes_for(task: str) -> list[str]:
    return {
        "dx3": ["CN", "MCI", "AD"],
        "adcn": ["CN", "AD"],
        "conv36": ["sMCI", "pMCI"],
    }[task]


def target_for(task: str) -> str:
    return "conv_36" if task == "conv36" else "baseline_dx"


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


def pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median", keep_empty_features=True),
            ),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=4000,
                    class_weight="balanced",
                    random_state=SEED,
                ),
            ),
        ]
    )


def aligned_probabilities(
    model: Pipeline, features: pd.DataFrame, classes: list[str]
) -> np.ndarray:
    probabilities = model.predict_proba(features)
    fitted_classes = list(model.named_steps["classifier"].classes_)
    if set(fitted_classes) != set(classes):
        raise RuntimeError(
            f"Classifier classes {fitted_classes} do not match {classes}"
        )
    return probabilities[:, [fitted_classes.index(name) for name in classes]]


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


def prediction_frame(
    task: str,
    fold: int,
    feature_set: str,
    rids: np.ndarray,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: list[str],
    signature: str,
) -> pd.DataFrame:
    predicted = np.asarray(classes)[probabilities.argmax(axis=1)]
    frame = pd.DataFrame(
        {
            "run_signature": signature,
            "task": task,
            "fold": fold,
            "featureset": feature_set,
            "RID": rids.astype(int),
            "y_true": y_true.astype(str),
            "y_pred": predicted,
        }
    )
    for index, class_name in enumerate(classes):
        frame[f"p_{class_name}"] = probabilities[:, index]
    return frame


def validate_fold_file(
    frame: pd.DataFrame,
    task: str,
    fold: int,
    classes: list[str],
    expected_signature: str | None,
) -> tuple[list[str], str]:
    probability_columns = [f"p_{name}" for name in classes]
    embedding_columns = sorted(
        column for column in frame.columns if column.startswith("emb_")
    )
    required = {
        "run_signature",
        "task",
        "fold",
        "split",
        "RID",
        "y_true",
        *probability_columns,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{task} fold{fold}: missing columns {sorted(missing)}"
        )
    if len(embedding_columns) != 512:
        raise ValueError(
            f"{task} fold{fold}: expected 512 embeddings, "
            f"found {len(embedding_columns)}"
        )
    if set(frame["task"].astype(str)) != {task}:
        raise ValueError(f"{task} fold{fold}: task column mismatch")
    if set(pd.to_numeric(frame["fold"], errors="raise").astype(int)) != {fold}:
        raise ValueError(f"{task} fold{fold}: fold column mismatch")
    signatures = set(frame["run_signature"].astype(str))
    if len(signatures) != 1:
        raise ValueError(f"{task} fold{fold}: multiple run signatures")
    signature = next(iter(signatures))
    if expected_signature is not None and signature != expected_signature:
        raise ValueError(
            f"{task} fold{fold}: run signature differs across folds"
        )
    if frame["RID"].duplicated().any():
        raise ValueError(f"{task} fold{fold}: duplicate RIDs")
    frame["RID"] = pd.to_numeric(frame["RID"], errors="raise").astype(int)
    split_values = set(frame["split"].astype(str))
    if split_values != {"train", "test"}:
        raise ValueError(
            f"{task} fold{fold}: expected train/test, found {split_values}"
        )
    train_rids = set(frame.loc[frame["split"] == "train", "RID"])
    test_rids = set(frame.loc[frame["split"] == "test", "RID"])
    if train_rids & test_rids:
        raise ValueError(f"{task} fold{fold}: train/test RID overlap")
    if not set(frame["y_true"].astype(str)).issubset(set(classes)):
        raise ValueError(f"{task} fold{fold}: unexpected labels")
    numeric = frame[probability_columns + embedding_columns].to_numpy(
        dtype=float
    )
    if not np.isfinite(numeric).all():
        raise ValueError(f"{task} fold{fold}: non-finite deep features")
    if not np.allclose(
        frame[probability_columns].sum(axis=1).to_numpy(), 1.0, atol=1e-4
    ):
        raise ValueError(f"{task} fold{fold}: probabilities do not sum to one")
    return embedding_columns, signature


def load_inputs(data_directory: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    master_path = data_directory / "master_features.csv"
    families_path = data_directory / "feature_families.json"
    if not master_path.is_file() or not families_path.is_file():
        raise FileNotFoundError(
            f"Required data files not found: {master_path} / {families_path}"
        )
    master = pd.read_csv(master_path, low_memory=False)
    if master["RID"].duplicated().any():
        raise ValueError("master_features.csv contains duplicate RIDs")
    master["RID"] = pd.to_numeric(master["RID"], errors="raise").astype(int)
    with families_path.open(encoding="utf-8") as handle:
        families = json.load(handle)
    clinical = [
        column
        for column in families["demo"] + families["cognition"]
        if column in master.columns
    ]
    freesurfer = [
        column for column in families["freesurfer"] if column in master.columns
    ]
    if not clinical:
        raise ValueError("No clinical features found in master_features.csv")
    if not freesurfer:
        raise ValueError("No FreeSurfer features found in master_features.csv")
    return master, clinical, freesurfer


def main() -> None:
    args = parse_args()
    if not args.results.is_dir():
        raise FileNotFoundError(
            f"Fixed training results not found: {args.results}"
        )
    master, clinical_columns, freesurfer_columns = load_inputs(args.data)
    all_predictions: list[pd.DataFrame] = []

    for task in TASKS:
        classes = classes_for(task)
        target = target_for(task)
        task_signature: str | None = None
        task_test_rids: list[int] = []

        for fold in range(FOLDS):
            feature_path = (
                args.results / f"deep_features_{task}_fold{fold}.csv"
            )
            if not feature_path.is_file():
                raise FileNotFoundError(
                    f"Missing fold features: {feature_path}. "
                    "Run train_cnn_cv.py first."
                )
            deep = pd.read_csv(feature_path, low_memory=False)
            embedding_columns, signature = validate_fold_file(
                deep, task, fold, classes, task_signature
            )
            task_signature = signature

            train_deep = deep[deep["split"] == "train"].copy()
            test_deep = deep[deep["split"] == "test"].copy()
            train = master.merge(
                train_deep, on="RID", how="inner", validate="one_to_one"
            )
            test = master.merge(
                test_deep, on="RID", how="inner", validate="one_to_one"
            )
            if len(train) != len(train_deep):
                missing = sorted(set(train_deep["RID"]) - set(train["RID"]))
                raise ValueError(
                    f"{task} fold{fold}: master missing train RIDs {missing[:20]}"
                )
            if len(test) != len(test_deep):
                missing = sorted(set(test_deep["RID"]) - set(test["RID"]))
                raise ValueError(
                    f"{task} fold{fold}: master missing test RIDs {missing[:20]}"
                )
            if set(train["RID"]) & set(test["RID"]):
                raise RuntimeError(f"{task} fold{fold}: RID leakage after merge")

            train_target = train[target].astype(str).to_numpy()
            test_target = test[target].astype(str).to_numpy()
            if not np.array_equal(
                train_target, train["y_true"].astype(str).to_numpy()
            ):
                raise ValueError(
                    f"{task} fold{fold}: train labels disagree with master"
                )
            if not np.array_equal(
                test_target, test["y_true"].astype(str).to_numpy()
            ):
                raise ValueError(
                    f"{task} fold{fold}: test labels disagree with master"
                )
            if set(train_target) != set(classes):
                raise ValueError(
                    f"{task} fold{fold}: outer-train lacks a class"
                )

            probability_columns = [f"p_{name}" for name in classes]
            deep_probabilities = test[probability_columns].to_numpy(dtype=float)
            all_predictions.append(
                prediction_frame(
                    task,
                    fold,
                    "deep_cnn",
                    test["RID"].to_numpy(),
                    test_target,
                    deep_probabilities,
                    classes,
                    signature,
                )
            )

            feature_sets = {
                "clinical": clinical_columns,
                "freesurfer": freesurfer_columns,
                "clinical+deep": clinical_columns + embedding_columns,
                "all": clinical_columns
                + freesurfer_columns
                + embedding_columns,
            }
            for feature_set, columns in feature_sets.items():
                estimator = pipeline()
                estimator.fit(train[columns], train_target)
                probabilities = aligned_probabilities(
                    estimator, test[columns], classes
                )
                all_predictions.append(
                    prediction_frame(
                        task,
                        fold,
                        feature_set,
                        test["RID"].to_numpy(),
                        test_target,
                        probabilities,
                        classes,
                        signature,
                    )
                )
            task_test_rids.extend(test["RID"].astype(int).tolist())

        if len(task_test_rids) != len(set(task_test_rids)):
            raise RuntimeError(f"{task}: a subject appears in multiple test folds")

    predictions = pd.concat(all_predictions, ignore_index=True)
    summary_rows: list[dict[str, object]] = []
    for task in TASKS:
        classes = classes_for(task)
        probability_columns = [f"p_{name}" for name in classes]
        expected_rids: set[int] | None = None
        for feature_set in FEATURESETS:
            selected = predictions[
                (predictions["task"] == task)
                & (predictions["featureset"] == feature_set)
            ].copy()
            if selected.empty:
                raise RuntimeError(f"No predictions for {task}/{feature_set}")
            if selected["RID"].duplicated().any():
                raise RuntimeError(f"Duplicate OOF RIDs for {task}/{feature_set}")
            current_rids = set(selected["RID"].astype(int))
            if expected_rids is None:
                expected_rids = current_rids
            elif current_rids != expected_rids:
                raise RuntimeError(
                    f"OOF cohorts differ within task {task}: {feature_set}"
                )
            y_true = selected["y_true"].astype(str).to_numpy()
            probabilities = selected[probability_columns].to_numpy(dtype=float)
            bacc = balanced_accuracy_score(
                y_true, selected["y_pred"].astype(str).to_numpy()
            )
            auc = auc_score(y_true, probabilities, classes)
            summary_rows.append(
                {
                    "task": task,
                    "featureset": feature_set,
                    "bAcc": float(bacc),
                    "auc": float(auc),
                    "n": len(selected),
                    "cv": "aligned_outer_5fold",
                }
            )
            print(
                f"  {task:<7} {feature_set:<14} "
                f"bAcc={bacc:.3f} AUC={auc:.3f} (n={len(selected)})"
            )

    summary = pd.DataFrame(summary_rows)
    summary_path = Path(f"{args.output_prefix}_summary.csv")
    oof_path = Path(f"{args.output_prefix}_oof.csv")
    figure_path = Path(f"{args.output_prefix}_comparison.png")
    atomic_csv(summary, summary_path)
    atomic_csv(
        predictions.sort_values(["task", "featureset", "RID"]).reset_index(
            drop=True
        ),
        oof_path,
    )

    order = list(FEATURESETS)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    chance_levels = (1 / 3, 0.5, 0.5)
    for axis, task, chance in zip(axes, TASKS, chance_levels):
        task_summary = summary[summary["task"] == task].set_index("featureset")
        values = [float(task_summary.loc[name, "bAcc"]) for name in order]
        axis.bar(
            order,
            values,
            color=["#2c7fb8", "#f0a500", "#756bb1", "#31a354", "#d7301f"],
        )
        axis.axhline(chance, linestyle="--", color="black")
        axis.set_ylim(0, 1.05)
        axis.set_title(task)
        axis.tick_params(axis="x", rotation=30)
        axis.set_ylabel("Balanced accuracy")
    figure.suptitle(
        "Leakage-safe CNN / clinical / FreeSurfer fusion (aligned outer 5-fold)"
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{figure_path.name}.", suffix=".png", dir=figure_path.parent
    )
    os.close(descriptor)
    temporary_figure = Path(temporary_name)
    try:
        figure.savefig(
            temporary_figure, dpi=150, bbox_inches="tight", format="png"
        )
        plt.close(figure)
        os.replace(temporary_figure, figure_path)
    finally:
        if temporary_figure.exists():
            temporary_figure.unlink()
        plt.close(figure)

    print("Saved:", summary_path)
    print("Saved:", oof_path)
    print("Saved:", figure_path)


if __name__ == "__main__":
    main()
