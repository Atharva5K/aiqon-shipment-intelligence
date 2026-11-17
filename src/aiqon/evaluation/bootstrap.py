"""Paired test-set bootstrap resampling at the shipment level."""
from __future__ import annotations

import numpy as np


def shipment_bootstrap(records, variants, checkpoints, replicates: int, seed: int):
    """Resample shipment groups once per replicate and reuse IDs for all comparisons."""
    if replicates < 1:
        raise ValueError("replicates must be positive")
    shipment_ids = np.asarray(sorted(records["shipment_id"].unique()), dtype=object)
    rng = np.random.default_rng(seed)
    values = {variant: {cp: {"mae": [], "improvement_over_carry_forward": []}
                         for cp in checkpoints} for variant in variants}
    carry = "carry_forward"
    sampled_id_records = []
    for _ in range(replicates):
        sampled = rng.choice(shipment_ids, size=len(shipment_ids), replace=True)
        sampled_id_records.append(sampled.tolist())
        draw = records.set_index("shipment_id").loc[sampled].reset_index()
        for cp in checkpoints:
            part = draw[draw["checkpoint_type"] == cp]
            truth = part["actual"].to_numpy(dtype=float)
            carry_mae = float(np.mean(np.abs(part[carry].to_numpy(dtype=float) - truth)))
            for variant in variants:
                mae = float(np.mean(np.abs(part[variant].to_numpy(dtype=float) - truth)))
                values[variant][cp]["mae"].append(mae)
                values[variant][cp]["improvement_over_carry_forward"].append(carry_mae - mae)
    output = {"replicates": int(replicates), "unique_test_shipments": int(len(shipment_ids)),
              "seed": int(seed), "resampling_unit": "shipment_id; all checkpoint rows kept together",
              "paired_comparisons": True, "metrics": {}}
    for variant in variants:
        output["metrics"][variant] = {}
        for cp in checkpoints:
            output["metrics"][variant][cp] = {}
            for metric, draws in values[variant][cp].items():
                output["metrics"][variant][cp][metric] = {
                    "point_estimate": float(np.mean(draws)),
                    "ci_95_percent": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))],
                }
    return output, sampled_id_records

