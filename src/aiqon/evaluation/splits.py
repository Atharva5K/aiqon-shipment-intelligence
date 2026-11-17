"""Deterministic shipment-level stratified Phase 3 partitioning."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _logical_table_paths(root: Path, tables):
    paths = []
    for folder, name in tables:
        for suffix in ("parquet", "csv"):
            path = root / folder / f"{name}.{suffix}"
            if path.exists():
                paths.append(path)
                break
        else:
            raise FileNotFoundError(f"Missing Phase 3 dataset input: {folder}/{name}")
    return paths


def _hash_files(paths, base: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(base).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def dataset_fingerprints(root: str | Path) -> dict[str, str]:
    """Fingerprint every raw table and both processed predictor/target tables."""
    root = Path(root)
    raw = _logical_table_paths(root, [("data/raw", name) for name in
        ("shipments", "events", "environment_actuals", "forecasts")])
    processed = _logical_table_paths(root, [("data/processed", name) for name in
        ("prediction_snapshots", "snapshot_targets")])
    return {"raw_dataset_fingerprint_sha256": _hash_files(raw, root),
            "processed_dataset_fingerprint_sha256": _hash_files(processed, root),
            "dataset_fingerprint_sha256": _hash_files([*raw, *processed], root)}


def _resolve_input_path(path: str | Path, root: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path.resolve()
    workspace_candidate = Path.cwd() / path
    if workspace_candidate.exists():
        return workspace_candidate.resolve()
    return (root / path).resolve()


def experiment_fingerprints(root: str | Path, simulation_config_path: str | Path,
                            routes_config_path: str | Path, model_configuration: dict,
                            random_seed: int) -> dict:
    """Hash data, configuration, routes, model settings and seed into provenance."""
    root = Path(root).resolve()
    data = dataset_fingerprints(root)
    config_path = _resolve_input_path(simulation_config_path, root)
    route_path = _resolve_input_path(routes_config_path, root)
    config_fingerprint = hashlib.sha256(config_path.read_bytes()).hexdigest()
    route_fingerprint = hashlib.sha256(route_path.read_bytes()).hexdigest()
    payload = {**data, "simulation_config_fingerprint_sha256": config_fingerprint,
               "route_config_fingerprint_sha256": route_fingerprint,
               "effective_phase3_model_configuration": model_configuration,
               "random_seed": int(random_seed)}
    experiment_fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    return {**payload, "experiment_fingerprint_sha256": experiment_fingerprint}


def dataset_fingerprint(root: str | Path) -> str:
    """Compatibility helper returning the full raw+processed dataset fingerprint."""
    return dataset_fingerprints(root)["dataset_fingerprint_sha256"]


def labels_by_shipment(shipments: pd.DataFrame, targets: pd.DataFrame) -> dict[str, int]:
    """Read one consistent significant-delay label per shipment, before features."""
    grouped = targets.groupby("shipment_id")["significant_delay_label"]
    inconsistent = grouped.nunique(dropna=False)
    bad = inconsistent[inconsistent != 1]
    if not bad.empty:
        raise ValueError(f"Significant-delay label must be consistent across checkpoints: {bad.index.tolist()[:5]}")
    labels = {str(sid): int(bool(values.iloc[0])) for sid, values in grouped}
    shipment_ids = {str(sid) for sid in shipments["shipment_id"]}
    if set(labels) != shipment_ids:
        raise ValueError("Shipment master IDs and target shipment IDs do not match")
    counts = targets.groupby("shipment_id").size()
    if not (counts == 3).all():
        raise ValueError("Each shipment must have exactly three target checkpoints")
    return labels


def stratified_shipment_split(labels: dict[str, int], seed: int,
                              fractions=(0.70, 0.15, 0.15)) -> dict[str, list[str]]:
    """Split unique IDs into exact 210/45/45 groups, failing if stratification cannot."""
    ids = np.asarray(sorted(labels), dtype=object)
    y = np.asarray([labels[sid] for sid in ids], dtype=int)
    if len(ids) != 300:
        raise ValueError(f"Phase 3 requires exactly 300 unique shipments; found {len(ids)}")
    if tuple(fractions) != (0.70, 0.15, 0.15):
        raise ValueError("Phase 3 shipment split is fixed at 70/15/15")
    if len(np.unique(y)) < 2:
        raise ValueError("Cannot stratify the fixed split: shipment labels contain only one class")
    try:
        development, test = train_test_split(
            ids, test_size=45, random_state=seed, stratify=y,
        )
        development_y = np.asarray([labels[sid] for sid in development], dtype=int)
        fit, calibration = train_test_split(
            development, test_size=45, random_state=seed + 1, stratify=development_y,
        )
    except ValueError as exc:
        raise ValueError(f"Cannot form the exact stratified 70/15/15 shipment split: {exc}") from exc
    result = {"fit": sorted(map(str, fit)), "calibration": sorted(map(str, calibration)),
              "test": sorted(map(str, test))}
    expected = {"fit": 210, "calibration": 45, "test": 45}
    if {name: len(values) for name, values in result.items()} != expected:
        raise AssertionError("Stratified split did not produce exact 210/45/45 shipment counts")
    all_ids = [sid for group in result.values() for sid in group]
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != set(labels):
        raise AssertionError("Shipment split contains overlap or missing shipment IDs")
    return result


def split_manifest(partitions: dict[str, list[str]], labels: dict[str, int],
                   fingerprint: str, seed: int, config: dict,
                   provenance: dict | None = None) -> dict:
    counts = {}
    for name, ids in partitions.items():
        counts[name] = {
            "shipments": len(ids),
            "checkpoint_snapshots": len(ids) * 3,
            "significant_delay_shipments": int(sum(labels[sid] for sid in ids)),
            "not_significant_shipments": int(len(ids) - sum(labels[sid] for sid in ids)),
        }
    return {
        "split_seed": int(seed), "dataset_fingerprint_sha256": fingerprint,
        "fit_shipment_ids": partitions["fit"],
        "calibration_shipment_ids": partitions["calibration"],
        "test_shipment_ids": partitions["test"],
        "partition_counts": counts,
        "checkpoint_snapshot_counts": {key: value["checkpoint_snapshots"] for key, value in counts.items()},
        "significant_delay_class_counts": {
            key: {"significant": value["significant_delay_shipments"],
                  "not_significant": value["not_significant_shipments"]}
            for key, value in counts.items()
        },
        "split_configuration": config,
        "experiment_provenance": provenance or {},
        "experiment_fingerprint_sha256": (provenance or {}).get("experiment_fingerprint_sha256"),
    }

