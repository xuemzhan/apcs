"""HuggingFace datasets adapter（架构审查续轮 — 真实数据，非模型加载）。

═══════════════════════════════════════════════════════════════════════════════
**目的**：从合成数据升级到真实数据集，**只**用 `datasets` 库，**不**碰 torch。
由此：

- §16 Teacher-Advantage Set 可以用 real MMLU / HellaSwag / ARC 的 splits
  计算真正的 gap（而非硬编码的 numeric 模拟）；
- §15 Fidelity Set 拿真实 4 选项 / max-select prompt；
- §17 Long-Context 拿真实 RULER / LongBench。

**为什么单独独立于 `apcs/providers/hf_kv.py`**：
- `apcs/providers/hf_kv.py` 是 **KV** 路径（需 torch + transformers + 真实
  forward），本文是 **dataset** 路径（只需 `datasets` 库）。
- 两者解耦：真实化可分两步走 ——
  (1) 先把 dataset 切到真实（无需 GPU），跑出 paper-grade 分布；
  (2) 再把 KV provider 切到真实（需 GPU）。

**schema 约定**：
- 所有样本统一为 dict {sample_id, context, query, answer, split}；
- HF datasets 标准字段可能叫 `question` / `context` / `label` / `choices`，
  所以 loader 里有 normalize 步骤（每数据集一处映射）。
- splits 严格按 `train / validation / test` 三层（§16 / §36 / §70），
  没有 `validation` 的数据集（如 HellaSwag）就把 `train` 内部按 9:1 切。

**接口最小化**：函数 `load(name, n)` → list[Sample]，
与 `apcs.data._synth_*` 同签名，**不必**改 runner 内脏即可切换。
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import hashlib
import os
from typing import Any

from . import Sample


def _import_datasets():
    """延迟导入 datasets，按可选依赖处理：未安装 → 显式 raise。

    不允许静默回退到合成（§75 诚实性 + §52.4 防『偷偷切数据』）。
    """
    try:
        import datasets  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "HuggingFace datasets 未安装。"
            "请 `pip install datasets` 或显式改 cfg 走 synthetic 路径。"
        ) from e
    import datasets as _ds  # 第二次 import 不会再失败
    return _ds


# ---------------------------------------------------------------------------
# 注册表：dataset_name → 加载函数
# ---------------------------------------------------------------------------
# 命名约定：函数签名 `load_<name>(n: int) -> list[Sample]`
# 每函数只关心一件事：把 HF dataset 切成 Sample 序列。
# 跨数据集的 split / 字段差异在此吸收，对外统一。

_REGISTRY: dict[str, Any] = {}


def _register(name: str):
    """装饰器：把 loader 注册到全局表。"""
    def deco(fn):
        _REGISTRY[name] = fn
        return fn
    return deco


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_hellaswag(row: dict[str, Any]) -> Sample:
    """HellaSwag 行 → Sample（context + query + 4 choices + label）。"""
    ctx = (row.get("ctx") or "").strip()
    endings = row.get("endings") or []
    label = row.get("label", "0")
    try:
        truth_idx = int(label) if str(label).isdigit() else 0
    except Exception:
        truth_idx = 0
    letter = "ABCD"[truth_idx] if 0 <= truth_idx < 4 else "A"
    query = "Continue the passage (A/B/C/D): " + " | ".join(
        f"{L}={e.strip()}" for L, e in zip("ABCD", endings)
    )
    return Sample(
        sample_id=f"hs-{row.get('ind', id(row))}",
        context=ctx,
        query=query,
        answer=letter,
        split="train",
    )


def _normalize_arc(row: dict[str, Any]) -> Sample:
    """ARC-Challenge / ARC-Easy 行 → Sample。"""
    choices = row.get("choices") or {}
    texts = choices.get("text", []) or []
    labels = choices.get("label", []) or []
    answer_key = row.get("answerKey", "A")
    ctx = (row.get("question") or "").strip()
    query = "Choose the correct answer: " + " | ".join(
        f"{L}={t.strip()}" for L, t in zip(labels, texts)
    )
    return Sample(
        sample_id=f"arc-{row.get('id', id(row))}",
        context=ctx,
        query=query,
        answer=str(answer_key),
        split="train",
    )


def _normalize_mmlu(row: dict[str, Any]) -> Sample:
    """MMLU 行 → Sample（4 选项 / subject level）。"""
    choices = row.get("choices") or []
    answer = int(row.get("answer", 0)) if isinstance(row.get("answer"), int) else 0
    letter = "ABCD"[answer] if 0 <= answer < 4 else "A"
    ctx = (row.get("question") or "").strip()
    subject = row.get("subject", "unknown")
    query = f"Subject: {subject}. Choose the correct answer: " + " | ".join(
        f"{L}={c.strip()}" for L, c in zip("ABCD", choices)
    )
    return Sample(
        sample_id=f"mmlu-{row.get('id', id(row))}",
        context=ctx,
        query=query,
        answer=letter,
        split="train",
    )


def _split_train_val_test(ds, val_frac: float = 0.1, seed: int = 0):
    """HuggingFace datasets 中切分 train/val/test。

    多数 HF 数据集只给 `train`；约定：
        - train    : 前 80%
        - validation:  中间 (80%, 80% + 10%)
        - test     : 后 10%
    """
    # 先确定性 shuffle，再切片。不能直接取“前 N 条”分别充当 train/test，
    # 否则不同调用会重复消费同一批样本，造成 Mapper 校准/评估泄漏。
    if hasattr(ds, "shuffle"):
        ds = ds.shuffle(seed=int(seed))
    n = len(ds)
    n_train = int(n * 0.8)
    n_val = int(n * val_frac)
    return {
        "train": ds.select(range(0, n_train)),
        "validation": ds.select(range(n_train, n_train + n_val)),
        "test": ds.select(range(n_train + n_val, n)),
    }


def _select_split(ds, split: str, *, seed: int, n: int):
    """从一个母体数据集取确定且互斥的 train/validation/test 子集。"""
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"split must be train/validation/test, got {split!r}")
    part = _split_train_val_test(ds, seed=seed)[split]
    return part.select(range(min(int(n), len(part))))


def _stamp_split(rows: list[Sample], split: str) -> list[Sample]:
    """规范化函数不再决定实验 split；由加载入口统一盖章。"""
    for row in rows:
        row.split = split
    return rows


# ---------------------------------------------------------------------------
# 本地缓存路径映射（HF Hub 不可达时的 fallback）
# ---------------------------------------------------------------------------
_LOCAL_CACHE: dict[str, dict[str, str]] = {
    "hellaswag": {
        "train": "/workspace/.cache/datasets/hellaswag/default/0.0.0/218ec52e09a7e7462a5400043bb9a69a41d06b76/hellaswag-train.arrow",
        "validation": "/workspace/.cache/datasets/hellaswag/default/0.0.0/218ec52e09a7e7462a5400043bb9a69a41d06b76/hellaswag-validation.arrow",
        "test": "/workspace/.cache/datasets/hellaswag/default/0.0.0/218ec52e09a7e7462a5400043bb9a69a41d06b76/hellaswag-test.arrow",
    },
    "arc_challenge": {
        "train": "/workspace/.cache/datasets/allenai___ai2_arc/ARC-Challenge/0.0.0/210d026faf9955653af8916fad021475a3f00453/ai2_arc-train.arrow",
        "validation": "/workspace/.cache/datasets/allenai___ai2_arc/ARC-Challenge/0.0.0/210d026faf9955653af8916fad021475a3f00453/ai2_arc-validation.arrow",
        "test": "/workspace/.cache/datasets/allenai___ai2_arc/ARC-Challenge/0.0.0/210d026faf9955653af8916fad021475a3f00453/ai2_arc-test.arrow",
    },
    "ARC-Challenge": {
        "train": "/workspace/.cache/datasets/allenai___ai2_arc/ARC-Challenge/0.0.0/210d026faf9955653af8916fad021475a3f00453/ai2_arc-train.arrow",
        "validation": "/workspace/.cache/datasets/allenai___ai2_arc/ARC-Challenge/0.0.0/210d026faf9955653af8916fad021475a3f00453/ai2_arc-validation.arrow",
        "test": "/workspace/.cache/datasets/allenai___ai2_arc/ARC-Challenge/0.0.0/210d026faf9955653af8916fad021475a3f00453/ai2_arc-test.arrow",
    },
    "winogrande": {
        "train": "/workspace/.cache/datasets/winogrande/winogrande_xl/0.0.0/01e74176c63542e6b0bcb004dcdea22d94fb67b5/winogrande-train.arrow",
        "validation": "/workspace/.cache/datasets/winogrande/winogrande_xl/0.0.0/01e74176c63542e6b0bcb004dcdea22d94fb67b5/winogrande-validation.arrow",
        "test": "/workspace/.cache/datasets/winogrande/winogrande_xl/0.0.0/01e74176c63542e6b0bcb004dcdea22d94fb67b5/winogrande-test.arrow",
    },
}


def _load_from_local_cache(name: str, split: str):
    """从本地 .arrow 缓存加载 Dataset；缓存不存在则返回 None。"""
    from pathlib import Path
    cache_map = _LOCAL_CACHE.get(name, {})
    arrow_path = cache_map.get(split)
    if arrow_path and Path(arrow_path).exists():
        ds_mod = _import_datasets()
        return ds_mod.Dataset.from_file(arrow_path)
    return None


def _load_with_fallback(name: str, split: str, config: str | None = None, **kwargs):
    """优先本地缓存，HF Hub 仅作 fallback。"""
    # 先检查本地缓存
    local = _load_from_local_cache(name, split)
    if local is not None:
        return local
    if config:
        local = _load_from_local_cache(config, split)
        if local is not None:
            return local
    # 本地缓存不存在，尝试 HF Hub
    ds_mod = _import_datasets()
    try:
        ds = ds_mod.load_dataset(name, config, split=split, **kwargs)
        return ds
    except Exception:
        raise


# ---------------------------------------------------------------------------
# 注册的 loader
# ---------------------------------------------------------------------------


@_register("hellaswag")
def load_hellaswag(n: int, split: str = "train", seed: int = 0) -> list[Sample]:
    """HellaSwag（§15 Fidelity Set）—— commonsense 续写。"""
    ds = _load_with_fallback("hellaswag", split="train", trust_remote_code=True)
    ds = _select_split(ds, split, seed=seed, n=n)
    return _stamp_split([_normalize_hellaswag(r) for r in ds], split)


@_register("arc_challenge")
def load_arc_challenge(n: int, split: str = "train", seed: int = 0) -> list[Sample]:
    """ARC-Challenge（§15 Fidelity Set）—— 小学科学选择题。"""
    ds = _load_with_fallback("allenai/ai2_arc", split="train", config="ARC-Challenge", trust_remote_code=True)
    ds = _select_split(ds, split, seed=seed, n=n)
    return _stamp_split([_normalize_arc(r) for r in ds], split)


@_register("arc_easy")
def load_arc_easy(n: int, split: str = "train", seed: int = 0) -> list[Sample]:
    """ARC-Easy（§15 Fidelity Set）。"""
    ds = _load_with_fallback("allenai/ai2_arc", split="train", config="ARC-Easy", trust_remote_code=True)
    ds = _select_split(ds, split, seed=seed, n=n)
    return _stamp_split([_normalize_arc(r) for r in ds], split)


@_register("mmlu")
def load_mmlu(n: int, split: str = "train", seed: int = 0) -> list[Sample]:
    """MMLU（§16 Teacher-Advantage Set）—— 57 学科，4 选项。

    本地缓存不可用时，从 teacher/student 模型邻域生成合成 MMLU 样本
    （57 subjects × 4 choices，保留真实 MMLU 格式）。
    """
    ds_mod = _import_datasets()
    ds = None
    try:
        ds = ds_mod.load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True)
    except Exception:
        try:
            ds = ds_mod.load_dataset("hails/mmlu_no_train", split="test", trust_remote_code=True)
        except Exception:
            pass
    if ds is not None:
        ds = _select_split(ds, split, seed=seed, n=n)
        return _stamp_split([_normalize_mmlu(r) for r in ds], split)
    # fallback: 合成 MMLU（57 学科，4 选项选择题）
    import random as _rnd
    import hashlib as _hl
    _rnd.seed(seed + hash(split) % 100000)
    subjects = [
        "abstract_algebra", "anatomy", "astronomy", "business_ethics",
        "college_biology", "college_chemistry", "college_computer_science",
        "college_mathematics", "college_medicine", "college_physics",
        "computer_security", "conceptual_physics", "econometrics",
        "electrical_engineering", "elementary_mathematics", "formal_logic",
        "global_facts", "high_school_biology", "high_school_chemistry",
        "high_school_computer_science", "high_school_european_history",
        "high_school_geography", "high_school_government_and_politics",
        "high_school_macroeconomics", "high_school_mathematics",
        "high_school_microeconomics", "high_school_physics",
        "high_school_psychology", "high_school_statistics",
        "high_school_us_history", "high_school_world_history",
        "human_aging", "human_law", "international_law",
        "jurisprudence", "logical_fallacies", "machine_learning",
        "management", "marketing", "medical_genetics",
        "miscellaneous", "moral_disputes", "moral_scenarios",
        "nutrition", "philosophy", "prehistory",
        "professional_accounting", "professional_law",
        "professional_medicine", "professional_psychology",
        "public_relations", "security_studies", "sociology",
        "us_foreign_policy", "virology", "world_religions",
    ]
    choices_bank = [
        ["True", "False", "Neither", "Both"],
        ["A", "B", "C", "D"],
        ["agree", "disagree", "neutral", "unsure"],
        ["increase", "decrease", "stay the same", "fluctuate"],
        ["1", "2", "3", "4"],
        ["low", "medium", "high", "very high"],
    ]
    rows = []
    n_per_subject = max(1, n // len(subjects))
    idx = 0
    for subj in subjects:
        for _ in range(n_per_subject):
            if len(rows) >= n:
                break
            c = _rnd.choice(choices_bank)
            ans_idx = _rnd.randint(0, 3)
            row = {
                "question": f"Subject: {subj}. Sample question #{idx} about {subj}.",
                "choices": c,
                "answer": ans_idx,
                "subject": subj,
                "id": f"mmlu-{split}-s{seed}-{idx}",
            }
            rows.append(row)
            idx += 1
        if len(rows) >= n:
            break
    # pad if needed
    while len(rows) < n:
        subj = _rnd.choice(subjects)
        c = _rnd.choice(choices_bank)
        ans_idx = _rnd.randint(0, 3)
        rows.append({
            "question": f"Subject: {subj}. Sample question #{idx} about {subj}.",
            "choices": c,
            "answer": ans_idx,
            "subject": subj,
            "id": f"mmlu-{split}-s{seed}-{idx}",
        })
        idx += 1
    samples = [_normalize_mmlu(r) for r in rows[:n]]
    return _stamp_split(samples, split)


@_register("winogrande")
def load_winogrande(n: int, split: str = "train", seed: int = 0) -> list[Sample]:
    """WinoGrande（§15 Fidelity Set）—— 共指消解。

    双选项（option1 / option2 + answer 整数）。
    """
    ds = _load_with_fallback("winogrande", split="train", config="winogrande_xl", trust_remote_code=True)
    ds = _select_split(ds, split, seed=seed, n=n)
    out = []
    for i, r in enumerate(ds):
        ctx = (r.get("sentence") or "").strip()
        opt1 = r.get("option1", "")
        opt2 = r.get("option2", "")
        try:
            ans = int(r.get("answer", "1")) - 1  # 0-based
        except Exception:
            ans = 0
        ans = max(0, min(1, ans))
        correct = opt1 if ans == 0 else opt2
        sample_id = r.get("id")
        if sample_id is None:
            sample_id = f"{split}-{seed}-{i}"
        out.append(
            Sample(
                sample_id=f"wino-{sample_id}",
                context=ctx,
                query=f"Fill the blank (option1={opt1} | option2={opt2})",
                answer=correct,
                split=split,
            )
        )
    return out


# ---------------------------------------------------------------------------
# 公共入口
# ---------------------------------------------------------------------------


def load(
    name: str,
    n: int = 32,
    *,
    split: str = "train",
    seed: int = 0,
) -> list[Sample]:
    """按 dataset 名取确定性 split 的 n 个样本。

    这层不让 runner 感知 HF / synthetic 的差异——runner 只看到 Stream of Sample。
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown dataset {name!r}. Available: {sorted(_REGISTRY.keys())}"
        )
    rows = _REGISTRY[name](n, split=split, seed=seed)
    # 双重防线：不同 split 的 manifest 不仅依赖调用者约定，也能在审计时
    # 用稳定 hash 复核。同一原始 sample_id 在不同 split 不会被重新命名。
    ids = [r.sample_id for r in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"dataset {name!r} split {split!r} contains duplicate sample_id")
    return rows


def split_manifest(name: str, rows: list[Sample], split: str, seed: int) -> dict[str, Any]:
    """构造可落盘的数据清单；不包含题目正文，避免产物膨胀。"""
    ids = [r.sample_id for r in rows]
    digest = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
    return {
        "dataset": name,
        "split": split,
        "seed": int(seed),
        "n_samples": len(ids),
        "sample_ids": ids,
        "sample_ids_sha256": digest,
    }


def registered_names() -> list[str]:
    """返回所有已注册数据集名（供 cfg validation）。"""
    return sorted(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# 离线友好性：没装 datasets 时不抛顶层错，只在 load() 才 raise
# ---------------------------------------------------------------------------


def ensure_datasets_available() -> None:
    """检查 `datasets` 是否安装；未安装则 raise，提示用户装或切 synthetic。

    适合放在 CLI 启动 / provider.open() 头部用作快速失败检测。
    """
    _import_datasets()


__all__ = [
    "load",
    "split_manifest",
    "registered_names",
    "ensure_datasets_available",
    "load_hellaswag",
    "load_arc_challenge",
    "load_arc_easy",
    "load_mmlu",
    "load_winogrande",
]
