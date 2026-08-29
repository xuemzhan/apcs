"""KV Cache / mapper 参数磁盘持久化（核心目标："KV Cache 持久化后经 mapper 迁移"）。

════════════════════════════════════════════════════════════════════════════════
两阶段部署语义（对应核心目标的"持久化 → 迁移 → 冷启动"）：

    离线阶段（offline，需要 Teacher GPU）：
        Teacher 跑用户数据 → 逐样本捕获 KV → save_kv_cache() 落盘
        → mapper 训练后 save_mapper_params() 落盘
        （teacher_full 分数一并落盘，在线阶段免加载 Teacher）

    在线阶段（online，只需 Student GPU）：
        load_store() 读盘 → （mapper 参数已持久化则直接加载）
        → transform → 注入 Student → 零 prefill 解码

存储布局（一个 store 目录）：
    <store_dir>/
        manifest.json          — 协议版本 / git hash / 模型对 / 校验和
        rows.json              — 评估样本（sample_id/context/query/answer）
        teacher_scores.json    — 离线阶段测得的 teacher_full 分数
        kv/<sample_id>.npz     — k,v (L,S,H,D) float32 + S + 可选 positions
        mapper_params.npz      — mapper.W（或 A/B）参数，键含 kv_kind

§75 诚实性：manifest 记录每个 npz 的 sha256；load 时校验形状与校验和，
不一致显式 raise —— 防止"读错缓存"这类静默错误。
════════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

STORE_VERSION = 1


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_kv_sample(
    store_dir: Path,
    sample_id: str,
    k: np.ndarray,
    v: np.ndarray,
    positions: np.ndarray | None = None,
    dtype: str = "float32",
) -> Path:
    """单样本 KV 落盘：<store_dir>/kv/<safe_id>.npz。

    k/v 形状 (L, S, H, D)；dtype 可选 float32/float16（float16 减半磁盘体积，
    注入前再转回 float32 —— 精度损失在 KV 的可接受范围内，由调用方选择）。
    """
    kv_dir = Path(store_dir) / "kv"
    kv_dir.mkdir(parents=True, exist_ok=True)
    safe_id = sample_id.replace("/", "_")
    k_arr = np.asarray(k, dtype=dtype)
    v_arr = np.asarray(v, dtype=dtype)
    payload: dict[str, Any] = {
        "k": k_arr,
        "v": v_arr,
        "seq_len": np.int64(k_arr.shape[1]),
        "sample_id": np.str_(sample_id),
    }
    if positions is not None:
        payload["positions"] = np.asarray(positions, dtype=np.float64)
    path = kv_dir / f"{safe_id}.npz"
    np.savez_compressed(path, **payload)
    return path


def load_kv_sample(store_dir: Path, sample_id: str) -> tuple[np.ndarray, np.ndarray, int]:
    """读单样本 KV → (k float32, v float32, S)。文件缺失或损坏 → raise。"""
    safe_id = sample_id.replace("/", "_")
    path = Path(store_dir) / "kv" / f"{safe_id}.npz"
    if not path.exists():
        raise FileNotFoundError(f"KV store 缺少样本 {sample_id}: {path}")
    with np.load(path, allow_pickle=False) as z:
        k = z["k"].astype(np.float32)
        v = z["v"].astype(np.float32)
        S = int(z["seq_len"])
    if k.shape[1] != S or v.shape != k.shape:
        raise ValueError(f"KV store 样本 {sample_id} 形状不一致: k={k.shape}, v={v.shape}, S={S}")
    return k, v, S


def save_mapper_params(store_dir: Path, mappers: dict[str, Any]) -> Path:
    """mapper 参数落盘。支持含 .W dict（RidgeMapper / RidgePerHeadMapper /
    TaskAwareRidgeMapper）与 .A/.B dict（LowRankMapper）的 mapper。

    mappers: {"K": mapper_k, "V": mapper_v}
    注意：§22 下 mapper.W 的键已含 kv_kind —— (kind, s, h)，因此扁平键
    直接用 "W|kind|s|h"，不再叠加 mappers dict 的前缀（避免双重前缀）。
    """
    payload: dict[str, np.ndarray] = {}
    for _kind, mapper in mappers.items():
        W = getattr(mapper, "W", None)
        if isinstance(W, dict) and W:
            for key, mat in W.items():
                payload[f"W|{'|'.join(map(str, key))}"] = np.asarray(mat)
        A = getattr(mapper, "A", None)
        B = getattr(mapper, "B", None)
        if isinstance(A, dict) and isinstance(B, dict) and A:
            for key, mat in A.items():
                payload[f"A|{'|'.join(map(str, key))}"] = np.asarray(mat)
            for key, mat in B.items():
                payload[f"B|{'|'.join(map(str, key))}"] = np.asarray(mat)
    if not payload:
        logger.warning("[kv-store] 无可持久化的 mapper 参数（类型不支持或未 fit）")
        return Path(store_dir) / "mapper_params.npz"
    path = Path(store_dir) / "mapper_params.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return path


def load_mapper_params(store_dir: Path, mappers: dict[str, Any]) -> bool:
    """从磁盘恢复 mapper 参数（就地写入 mapper.W / mapper.A/B）。

    返回 True 表示至少恢复了一个参数矩阵；False 表示磁盘上没有参数
    （调用方应回退到重新 fit，并在 provenance 中记录）。
    """
    path = Path(store_dir) / "mapper_params.npz"
    if not path.exists():
        return False
    restored = False
    with np.load(path, allow_pickle=False) as z:
        for flat_key in z.files:
            parts = flat_key.split("|")
            # 扁平键 "W|kind|s|h"（kind 已含在 §22 键里）
            if len(parts) < 4:
                continue
            field, kind = parts[0], parts[1]
            mapper = mappers.get(kind)
            if mapper is None:
                continue
            try:
                nums = [int(x) for x in parts[2:]]
            except ValueError:
                continue
            key = (kind, *nums)
            mat = z[flat_key]
            if field == "W" and isinstance(getattr(mapper, "W", None), dict):
                mapper.W[key] = mat
                restored = True
            elif field == "A" and isinstance(getattr(mapper, "A", None), dict):
                mapper.A[key] = mat
                restored = True
            elif field == "B" and isinstance(getattr(mapper, "B", None), dict):
                mapper.B[key] = mat
                restored = True
    return restored


def write_manifest(
    store_dir: Path,
    meta: dict[str, Any],
) -> Path:
    """写 manifest.json：协议版本 / 元信息 / 每个 npz 的 sha256 校验和。"""
    store_dir = Path(store_dir)
    manifest = {
        "store_version": STORE_VERSION,
        "meta": meta,
        "checksums": {},
    }
    kv_dir = store_dir / "kv"
    if kv_dir.exists():
        for p in sorted(kv_dir.glob("*.npz")):
            manifest["checksums"][str(p.relative_to(store_dir))] = _sha256_file(p)
    for extra in ("mapper_params.npz",):
        p = store_dir / extra
        if p.exists():
            manifest["checksums"][extra] = _sha256_file(p)
    path = store_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_rows_teacher(
    store_dir: Path,
    rows: list[dict[str, Any]],
    teacher_scores: dict[str, Any],
) -> None:
    """评估样本与离线 teacher 分数落盘（在线阶段免 Teacher 的输入）。"""
    store_dir = Path(store_dir)
    (store_dir / "rows.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (store_dir / "teacher_scores.json").write_text(
        json.dumps(teacher_scores, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_store(store_dir: Path, verify: bool = True) -> dict[str, Any]:
    """读整个 store：rows / kv / teacher_scores / mapper_params 存在性。

    verify=True 时校验 manifest 中记录的 sha256（不匹配 → raise，§75）。
    """
    store_dir = Path(store_dir)
    manifest_path = store_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"KV store 缺少 manifest.json: {store_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if verify:
        for rel, expected in manifest.get("checksums", {}).items():
            p = store_dir / rel
            if not p.exists():
                raise FileNotFoundError(f"KV store 校验失败，文件缺失: {p}")
            actual = _sha256_file(p)
            if actual != expected:
                raise ValueError(f"KV store 校验和不匹配: {rel}（文件可能被改动）")
    rows_path = store_dir / "rows.json"
    rows = json.loads(rows_path.read_text(encoding="utf-8")) if rows_path.exists() else []
    teacher_path = store_dir / "teacher_scores.json"
    teacher_scores = (
        json.loads(teacher_path.read_text(encoding="utf-8")) if teacher_path.exists() else {}
    )
    kv_ids = sorted(
        p.stem for p in (store_dir / "kv").glob("*.npz")
    ) if (store_dir / "kv").exists() else []
    return {
        "manifest": manifest,
        "rows": rows,
        "teacher_scores": teacher_scores,
        "kv_sample_ids": kv_ids,
        "has_mapper_params": (store_dir / "mapper_params.npz").exists(),
    }
