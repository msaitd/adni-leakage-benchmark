"""
Build the fixed deep-learning manifest from the exact baseline scans selected in
run_local/process_list.csv.

The script is intentionally strict: it refuses to write a manifest when a
baseline scan is missing, duplicated, unlabeled, or not paired as mwp1+mwp2.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import tempfile
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_PROJECT = HERE.parent
DEFAULT_DERIV = Path(r"F:\ADNI_derivatives\cat12")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deriv", type=Path, default=DEFAULT_DERIV)
    parser.add_argument("--data", type=Path, default=DEFAULT_PROJECT / "data")
    parser.add_argument(
        "--process-list",
        type=Path,
        default=DEFAULT_PROJECT / "run_local" / "process_list.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "fixed_deep_manifest.csv",
    )
    parser.add_argument("--expected-subjects", type=int, default=645)
    return parser.parse_args()


def parse_cat12_path(path: str) -> tuple[str | None, int | None]:
    ptid_match = re.search(r"(\d{3}_S_\d{4})", path)
    iid_match = re.search(r"_I(\d+)\.nii$", path, flags=re.IGNORECASE)
    ptid = ptid_match.group(1) if ptid_match else None
    iid = int(iid_match.group(1)) if iid_match else None
    return ptid, iid


def baseline_selection(path: Path, expected_subjects: int) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"process_list.csv not found: {path}")
    table = pd.read_csv(path, low_memory=False)
    required = {"image_id", "ptid", "is_baseline"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"process_list.csv missing columns: {sorted(missing)}")

    baseline_flag = table["is_baseline"].astype(str).str.strip().str.lower()
    table = table[baseline_flag.isin({"true", "1"})].copy()
    table["PTID"] = table["ptid"].astype(str).str.strip()
    table["ImageUID"] = pd.to_numeric(
        table["image_id"].astype(str).str.replace(r"^I", "", regex=True),
        errors="raise",
    ).astype(int)

    if len(table) != expected_subjects:
        raise ValueError(
            f"Expected {expected_subjects} baseline rows, found {len(table)}"
        )
    if table["PTID"].duplicated().any():
        duplicated = table.loc[table["PTID"].duplicated(False), "PTID"].tolist()
        raise ValueError(f"Duplicate baseline PTIDs: {duplicated[:10]}")
    if table["ImageUID"].duplicated().any():
        duplicated = table.loc[
            table["ImageUID"].duplicated(False), "ImageUID"
        ].tolist()
        raise ValueError(f"Duplicate baseline ImageUIDs: {duplicated[:10]}")
    invalid_ptid = ~table["PTID"].str.fullmatch(r"\d{3}_S_\d{4}")
    if invalid_ptid.any():
        raise ValueError(
            f"Invalid PTIDs: {table.loc[invalid_ptid, 'PTID'].tolist()[:10]}"
        )
    return table[["PTID", "ImageUID"]].reset_index(drop=True)


def scan_cat12_pairs(deriv: Path) -> dict[tuple[str, int], dict[str, str]]:
    if not deriv.is_dir():
        raise FileNotFoundError(f"CAT12 derivative directory not found: {deriv}")

    rows: dict[tuple[str, int], dict[str, str]] = {}
    duplicates: list[tuple[str, int, str, str, str]] = []
    for kind in ("mwp1", "mwp2"):
        pattern = os.path.join(str(deriv), "**", f"{kind}*.nii")
        for file_path in glob.iglob(pattern, recursive=True):
            ptid, iid = parse_cat12_path(file_path)
            if ptid is None or iid is None:
                continue
            key = (ptid, iid)
            existing = rows.setdefault(key, {}).get(kind)
            if existing and os.path.normcase(existing) != os.path.normcase(file_path):
                duplicates.append((ptid, iid, kind, existing, file_path))
            rows[key][kind] = os.path.abspath(file_path)

    if duplicates:
        first = duplicates[0]
        raise ValueError(
            "Multiple CAT12 maps found for the same scan/channel: "
            f"{first[0]} I{first[1]} {first[2]}\n{first[3]}\n{first[4]}"
        )
    return rows


def unique_ptid_to_rid(dx: pd.DataFrame) -> dict[str, int]:
    required = {"PTID", "RID"}
    missing = required - set(dx.columns)
    if missing:
        raise ValueError(f"dxsum.csv missing columns: {sorted(missing)}")
    mapping = dx[["PTID", "RID"]].dropna().copy()
    mapping["PTID"] = mapping["PTID"].astype(str)
    mapping["RID"] = pd.to_numeric(mapping["RID"], errors="raise").astype(int)
    conflicts = mapping.groupby("PTID")["RID"].nunique()
    conflicts = conflicts[conflicts != 1]
    if not conflicts.empty:
        raise ValueError(f"Conflicting PTID-to-RID mappings: {conflicts.index[:10].tolist()}")
    mapping = mapping.drop_duplicates("PTID")
    return dict(zip(mapping["PTID"], mapping["RID"]))


def atomic_to_csv(frame: pd.DataFrame, output: Path) -> None:
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


def build_manifest(args: argparse.Namespace) -> pd.DataFrame:
    selected = baseline_selection(args.process_list, args.expected_subjects)
    pairs = scan_cat12_pairs(args.deriv)

    missing_pairs: list[str] = []
    records: list[dict[str, object]] = []
    for row in selected.itertuples(index=False):
        key = (row.PTID, int(row.ImageUID))
        pair = pairs.get(key, {})
        missing_channels = [kind for kind in ("mwp1", "mwp2") if kind not in pair]
        if missing_channels:
            missing_pairs.append(
                f"{row.PTID} I{row.ImageUID}: {','.join(missing_channels)}"
            )
            continue
        records.append(
            {
                "PTID": row.PTID,
                "ImageUID": int(row.ImageUID),
                "path_mwp1": pair["mwp1"],
                "path_mwp2": pair["mwp2"],
            }
        )

    if missing_pairs:
        preview = "\n".join(missing_pairs[:20])
        raise ValueError(
            f"{len(missing_pairs)} selected baseline scans lack paired CAT12 maps:\n"
            f"{preview}"
        )

    manifest = pd.DataFrame(records)
    dx_path = args.data / "dxsum.csv"
    master_path = args.data / "master_features.csv"
    if not dx_path.is_file() or not master_path.is_file():
        raise FileNotFoundError(
            f"Required data files not found: {dx_path} / {master_path}"
        )

    dx = pd.read_csv(dx_path, low_memory=False)
    ptid_to_rid = unique_ptid_to_rid(dx)
    manifest["RID"] = manifest["PTID"].map(ptid_to_rid)
    if manifest["RID"].isna().any():
        missing_ptids = manifest.loc[manifest["RID"].isna(), "PTID"].tolist()
        raise ValueError(f"PTIDs without RID mapping: {missing_ptids[:20]}")
    manifest["RID"] = manifest["RID"].astype(int)

    master = pd.read_csv(master_path, low_memory=False)
    required_labels = {"RID", "baseline_dx", "conv_36"}
    missing_labels = required_labels - set(master.columns)
    if missing_labels:
        raise ValueError(
            f"master_features.csv missing columns: {sorted(missing_labels)}"
        )
    labels = master[["RID", "baseline_dx", "conv_36"]].copy()
    if labels["RID"].duplicated().any():
        raise ValueError("master_features.csv contains duplicate RIDs")
    manifest = manifest.merge(labels, on="RID", how="left", validate="one_to_one")

    if len(manifest) != args.expected_subjects:
        raise ValueError(
            f"Final manifest has {len(manifest)} rows; expected {args.expected_subjects}"
        )
    if manifest["RID"].duplicated().any():
        raise ValueError("Final manifest contains duplicate RIDs")
    if manifest["baseline_dx"].isna().any():
        missing_rids = manifest.loc[manifest["baseline_dx"].isna(), "RID"].tolist()
        raise ValueError(f"Subjects without baseline diagnosis: {missing_rids[:20]}")

    valid_dx = {"CN", "MCI", "AD"}
    unexpected = set(manifest["baseline_dx"].dropna().astype(str)) - valid_dx
    if unexpected:
        raise ValueError(f"Unexpected baseline diagnoses: {sorted(unexpected)}")

    manifest = manifest[
        [
            "PTID",
            "ImageUID",
            "path_mwp1",
            "path_mwp2",
            "RID",
            "baseline_dx",
            "conv_36",
        ]
    ].sort_values("RID").reset_index(drop=True)
    return manifest


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args)
    atomic_to_csv(manifest, args.output)
    print(f"fixed manifest: {len(manifest)} subjects")
    print("  baseline_dx:", manifest["baseline_dx"].value_counts().to_dict())
    print("  conv_36    :", manifest["conv_36"].value_counts(dropna=False).to_dict())
    print("saved ->", args.output)


if __name__ == "__main__":
    main()
