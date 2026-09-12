"""Held-out reconstruction R2 for the per-head affine mapper (P1-16 / round-3 review).

Measures how well a per-head affine map fitted on calibration contexts recovers
the student's own K/V from the teacher's K/V. The unit is a *held-out
calibration context*: for each held-out context we fit-scored a single R2
between predicted and true student states (pooled over layers, heads, and token
positions, following the repository's T04 ``_score_kv`` convention), then report
the mean over held-out contexts.

Output: ``reports/reconstruction_r2/reconstruction_r2.json`` plus stdout.

Usage:
    python scripts/measure_reconstruction_r2.py \
        --config reports/runs/v15-4b-to-1.7b-affine-c30-s42-20260829-231341/inject-eval/config.json \
        --n-calib 20 --n-eval 10 --seq 512
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="reports/reconstruction_r2/reconstruction_r2.json")
    ap.add_argument("--n-calib", type=int, default=20)
    ap.add_argument("--n-eval", type=int, default=10)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--mapper", default="affine")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    # Prefer locally downloaded modelscope snapshots when present.
    for role in ("teacher", "student"):
        local = Path(f"/workspace/models/{cfg[role]['model_id'].split('/')[-1]}")
        if not local.exists():
            # normalise casing of the Qwen3-<size> suffix
            stem = cfg[role]["model_id"].split("/")[-1]
            for cand in Path("/workspace/models").glob("*"):
                if cand.name.lower() == stem.lower():
                    local = cand
                    break
        if local.exists():
            cfg[role]["model_id"] = str(local)

    from apcs.alignment.runner import proportional_mapping
    from apcs.mapper.math import AffineMapper, RidgePerHeadMapper
    from apcs.mapper.runner import (
        _de_rope_for_kind,
        _real_kv_splits,
        _ridge_lambda,
        kv_kinds,
    )
    from apcs.rope.runner import _rope_pairs, de_rope

    run_dir = Path("reports/reconstruction_r2/_provider")
    run_dir.mkdir(parents=True, exist_ok=True)

    n_t = int(cfg["teacher"].get("num_layers", 36))
    n_s = int(cfg["student"].get("num_layers", 28))
    D = int(cfg["teacher"].get("head_dim", 128))
    layer_map = proportional_mapping(n_t, n_s)
    inv_freq = _rope_pairs(D, theta=1_000_000.0)
    de_rope_fn = lambda k, p: de_rope(k, p, inv_freq)  # noqa: E731

    splits, seq_stats = _real_kv_splits(
        cfg, run_dir, n_calib=args.n_calib, n_eval=args.n_eval,
        requested_seq=args.seq,
    )

    out: dict[str, object] = {
        "config": args.config,
        "mapper": args.mapper,
        "n_calib": args.n_calib,
        "n_eval": args.n_eval,
        "requested_seq": args.seq,
        "seq_stats": seq_stats,
        "unit": "held-out calibration context",
        "r2_aggregation": "pooled per context, then mean over contexts; per-(layer,head) macro-average also reported",
        "kinds": {},
    }

    def _r2_perhead(pred: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
        """Return (pooled_r2, per-(layer,head) macro-average r2)."""
        from apcs.metrics import r2 as _r2
        L, S, H, Dd = ref.shape
        per_head: list[float] = []
        for layer in range(L):
            for h in range(H):
                a = pred[layer, :, h, :].reshape(-1, Dd)
                b = ref[layer, :, h, :].reshape(-1, Dd)
                per_head.append(float(_r2(b, a)))
        pooled = float(_r2(ref.reshape(-1, Dd), pred.reshape(-1, Dd)))
        return pooled, float(np.mean(per_head))

    for kind in kv_kinds(cfg):
        mapper = AffineMapper(lam=_ridge_lambda(cfg, kind)) if args.mapper == "affine" \
            else RidgePerHeadMapper(lam=_ridge_lambda(cfg, kind))
        # Variant 1: follow the configured de-RoPE path (the mapper's actual input).
        kind_dr = _de_rope_for_kind(cfg, kind, de_rope_fn)
        mapper.fit_batch(
            list(splits[kind]["calib"]), layer_map, kv_kind=kind,
            positions=None, de_rope_fn=kind_dr,
        )
        # Variant 2: raw KV (no de-RoPE), for reference.
        mapper_raw = AffineMapper(lam=_ridge_lambda(cfg, kind)) if args.mapper == "affine" \
            else RidgePerHeadMapper(lam=_ridge_lambda(cfg, kind))
        mapper_raw.fit_batch(
            list(splits[kind]["calib"]), layer_map, kv_kind=kind,
            positions=None, de_rope_fn=None,
        )
        entries: dict[str, object] = {}
        for tag, mp, dr in (("de_rope", mapper, kind_dr), ("raw", mapper_raw, None)):
            pooled_list: list[float] = []
            perhead_list: list[float] = []
            for kv_t, kv_s in splits[kind]["eval"]:
                S = int(kv_t.shape[1])
                pred = mp.transform(
                    kv_t, layer_map, kv_kind=kind,
                    positions=np.arange(S, dtype=np.float64), de_rope_fn=dr,
                )
                pooled, perhead = _r2_perhead(pred, kv_s)
                pooled_list.append(pooled)
                perhead_list.append(perhead)
            entries[tag] = {
                "mean_r2_pooled": float(np.mean(pooled_list)),
                "mean_r2_per_layer_head": float(np.mean(perhead_list)),
                "pooled_r2_per_context": pooled_list,
                "per_layer_head_r2_per_context": perhead_list,
                "n_heldout": len(pooled_list),
            }
            print(f"[recon-r2] {kind}/{tag}: pooled={np.mean(pooled_list):+.4f}  "
                  f"per-(layer,head)={np.mean(perhead_list):+.4f} (n_heldout={len(pooled_list)})")
        out["kinds"][kind] = entries

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[recon-r2] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
