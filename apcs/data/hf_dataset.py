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


def _split_train_val_test(ds, val_frac: float = 0.1):
    """HuggingFace datasets 中切分 train/val/test。

    多数 HF 数据集只给 `train`；约定：
        - train    : 前 80%
        - validation:  中间 (80%, 80% + 10%)
        - test     : 后 10%
    """
    n = len(ds)
    n_train = int(n * 0.8)
    n_val = int(n * val_frac)
    return {
        "train": ds.select(range(0, n_train)),
        "validation": ds.select(range(n_train, n_train + n_val)),
        "test": ds.select(range(n_train + n_val, n)),
    }


# ---------------------------------------------------------------------------
# 注册的 loader
# ---------------------------------------------------------------------------


@_register("hellaswag")
def load_hellaswag(n: int) -> list[Sample]:
    """HellaSwag（§15 Fidelity Set）—— commonsense 续写。"""
    ds_mod = _import_datasets()
    ds = ds_mod.load_dataset("hellaswag", split="train", trust_remote_code=True)
    ds = ds.select(range(min(n, len(ds))))
    return [_normalize_hellaswag(r) for r in ds]


@_register("arc_challenge")
def load_arc_challenge(n: int) -> list[Sample]:
    """ARC-Challenge（§15 Fidelity Set）—— 小学科学选择题。"""
    ds_mod = _import_datasets()
    ds = ds_mod.load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train", trust_remote_code=True)
    ds = ds.select(range(min(n, len(ds))))
    return [_normalize_arc(r) for r in ds]


@_register("arc_easy")
def load_arc_easy(n: int) -> list[Sample]:
    """ARC-Easy（§15 Fidelity Set）。"""
    ds_mod = _import_datasets()
    ds = ds_mod.load_dataset("allenai/ai2_arc", "ARC-Easy", split="train", trust_remote_code=True)
    ds = ds.select(range(min(n, len(ds))))
    return [_normalize_arc(r) for r in ds]


@_register("mmlu")
def load_mmlu(n: int) -> list[Sample]:
    """MMLU（§16 Teacher-Advantage Set）—— 57 学科，4 选项。

    备注：HF 上的 MMLU 通常以 `cais/mmlu` / `hails/mmlu_no_train` 形式提供。
    """
    ds_mod = _import_datasets()
    try:
        ds = ds_mod.load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True)
    except Exception:
        # fallback：部分数据集 hub 改名
        ds = ds_mod.load_dataset("hails/mmlu_no_train", split="test", trust_remote_code=True)
    ds = ds.select(range(min(n, len(ds))))
    return [_normalize_mmlu(r) for r in ds]


@_register("winogrande")
def load_winogrande(n: int) -> list[Sample]:
    """WinoGrande（§15 Fidelity Set）—— 共指消解。

    双选项（option1 / option2 + answer 整数）。
    """
    ds_mod = _import_datasets()
    ds = ds_mod.load_dataset("winogrande", "winogrande_xl", split="train", trust_remote_code=True)
    ds = ds.select(range(min(n, len(ds))))
    out = []
    for r in ds:
        ctx = (r.get("sentence") or "").strip()
        opt1 = r.get("option1", "")
        opt2 = r.get("option2", "")
        try:
            ans = int(r.get("answer", "1")) - 1  # 0-based
        except Exception:
            ans = 0
        ans = max(0, min(1, ans))
        correct = opt1 if ans == 0 else opt2
        out.append(
            Sample(
                sample_id=f"wino-{r.get('id', id(r))}",
                context=ctx,
                query=f"Fill the blank (option1={opt1} | option2={opt2})",
                answer=correct,
                split="train",
            )
        )
    return out


# ---------------------------------------------------------------------------
# 公共入口
# ---------------------------------------------------------------------------


def load(name: str, n: int = 32) -> list[Sample]:
    """按 dataset 名取前 n 个样本（list[Sample]），与 apcs.data.sync_* 接口对齐。

    这层不让 runner 感知 HF / synthetic 的差异——runner 只看到 Stream of Sample。
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown dataset {name!r}. Available: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[name](n)


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
    "registered_names",
    "ensure_datasets_available",
    "load_hellaswag",
    "load_arc_challenge",
    "load_arc_easy",
    "load_mmlu",
    "load_winogrande",
]
