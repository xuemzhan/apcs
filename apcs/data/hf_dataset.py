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

from . import Sample, assert_xlevel_disjoint


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
    "mmlu": {
        "test": "/workspace/.cache/datasets/cais___mmlu/all/0.0.0/apcs_mmlu_all/mmlu-test.arrow",
    },
}


def _load_from_local_cache(name: str, split: str):
    """从本地 .arrow 缓存加载 Dataset；缓存不存在则返回 None。

    读取失败（缺文件 / Dataset 读取异常 / 测试环境假模块）时同样返回 None，
    由调用方回退到 HF Hub —— 本地缓存只是加速/离线回退，不是强约束。
    """
    from pathlib import Path
    cache_map = _LOCAL_CACHE.get(name, {})
    arrow_path = cache_map.get(split)
    if arrow_path and Path(arrow_path).exists():
        try:
            ds_mod = _import_datasets()
            return ds_mod.Dataset.from_file(arrow_path)
        except Exception:
            return None
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

    仅从 HF Hub 加载真实 MMLU 数据；
    cais/mmlu 与 hails/mmlu_no_train 均不可用时直接拒绝，不合成假数据
    （§75 诚实性 + §52.4 防『偷偷切数据』）。
    """
    ds_mod = _import_datasets()
    # 本地缓存优先（modelscope.cn 拉取的 cais/mmlu "all" → test 基线），
    # HF 仅作 fallback：cais/mmlu 与 hails/mmlu_no_train 均不可用时
    # 直接拒绝，不合成假数据（§75 诚实性 + §52.4 防『偷偷切数据』）。
    ds = _load_from_local_cache("mmlu", split="test")
    if ds is None:
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
    raise RuntimeError(
        "MMLU unavailable: cais/mmlu and hails/mmlu_no_train both failed to load; "
        "refusing to synthesize fake MMLU data"
    )


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
# §17 Long-Context Needle-in-a-Haystack 本地生成器
# ---------------------------------------------------------------------------

# 填充用段落池：生成近自然的叙事文本（不依赖 HF Hub）
_PARAGRAPHS: list[str] = [
    "The morning began with a gentle breeze that carried the scent of pine "
    "through the valley. Birds sang their familiar melodies from the treetops, "
    "and the sunlight painted golden patterns on the forest floor.",

    "In the bustling marketplace, vendors arranged their wares with practiced "
    "precision. Colorful fabrics hung from wooden stalls, and the aroma of "
    "fresh bread mingled with the earthy scent of hand-ground spices.",

    "The ancient library held thousands of volumes, each one carefully "
    "catalogued by subject and era. Dust motes danced in the narrow beams "
    "of light that filtered through the high arched windows above.",

    "Rain pattered steadily against the windowpane as the train wound its way "
    "through the countryside. Green hills rolled endlessly toward the horizon, "
    "dotted with white houses and clusters of dark evergreen trees.",

    "The laboratory was quiet except for the soft hum of equipment. Precise "
    "measurements required absolute concentration, and the scientist adjusted "
    "each dial with meticulous care before recording the observations.",

    "A small fishing boat bobbed on the morning tide. The fisherman cast his "
    "net into the deep blue water, watching as it spread in a wide arc before "
    "sinking slowly beneath the surface.",

    "The mountain trail grew steeper with each switchback. Wildflowers clung "
    "to the rocky soil beside the path, their bright colors standing out "
    "against the grey stone and dark green scrub brush.",

    "Inside the workshop, the sound of a hammer on metal rang out in a "
    "steady rhythm. Sparks flew with each precise strike as the blacksmith "
    "shaped the glowing iron bar on his heavy anvil.",

    "The concert hall filled slowly with the murmur of the arriving audience. "
    "Musicians tuned their instruments in the warm glow of the stage lights, "
    "preparing for the evening performance ahead.",

    "Snow fell silently over the sleeping village. Chimney smoke rose in "
    "lazy spirals against the pale grey sky, and the only sound was the "
    "occasional creak of a snow-laden branch giving way under its weight.",

    "The ship cut through the cold Atlantic waves on a clear winter morning. "
    "Ice crystals glittered on the railing, and the crew moved efficiently "
    "through their tasks despite the biting wind.",

    "A fox padded quietly through the underbrush at twilight. Its russet fur "
    "blended with the autumn leaves, and its ears swiveled forward at every "
    "small sound in the gathering darkness.",

    "The stone bridge spanned the narrow gorge where two rivers met. Below, "
    "the water churned white over the rocks, and the spray hung in the air "
    "like a fine mist that nourished the ferns along the bank.",

    "Candles flickered in the dim dining room as the family gathered around "
    "the heavy oak table. Steam rose from the soup bowls, and the warmth "
    "of the fire in the hearth pushed back the evening chill.",

    "The botanist knelt beside the rare orchid, sketching its intricate "
    "petals in her field notebook. Each specimen was numbered and GPS-tagged "
    "before being carefully pressed between sheets of absorbent paper.",
]


def _build_filler(target_words: int, context_idx: int) -> str:
    """构建确定性填充文本，约 target_words 词。

    从段落池循环取完整段落（自然度高），截到接近目标长度的句号处。
    words→tokens 启发式：~1.3 tokens/word（见 load_needle_longctx docstring）。
    """
    parts: list[str] = []
    word_count = 0
    pidx = context_idx % len(_PARAGRAPHS)
    while word_count < target_words:
        para = _PARAGRAPHS[pidx % len(_PARAGRAPHS)]
        parts.append(para)
        word_count += len(para.split())
        pidx += 1
    text = " ".join(parts)
    if word_count > target_words * 1.3:
        words = text.split()
        truncated = " ".join(words[:target_words])
        last_period = truncated.rfind(".")
        if last_period > len(truncated) // 2:
            text = truncated[: last_period + 1]
        else:
            text = truncated + "."
    return text


# 候选 key / value 对（needle facts）
_KEYS: list[str] = [
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta",
    "theta", "kappa", "lambda", "sigma", "omega", "phi",
]
_VALUES: list[int] = [
    42, 137, 256, 512, 1024, 2048, 3141, 8080, 9001, 6174, 2718, 1618,
]


@_register("needle_longctx")
def load_needle_longctx(
    n: int,
    split: str = "train",
    seed: int = 0,
    *,
    queries_per_context: int = 2,
    target_tokens: int = 1024,
) -> list[Sample]:
    """§17 Long-Context Needle-in-a-Haystack 离线生成器（不依赖 HF Hub）。

    每个上下文 X = filler 叙事文本 + 1..3 条嵌入式 needle facts，
    格式为 "The magic number for <key> is <value>."；
    每个 X 生成 queries_per_context 条 query（默认 2，可配2~4），
    例如 "What is the magic number for <key>?"，answer=<value>。

    上下文级分层（§16 §17）：
        所有来自同一 X 的 query 共享同一个 context_id，归入同一 split；
        不同 context_id 在 train/validation/test 之间互斥。
        验证通过 assert_xlevel_disjoint(rows) 保证。

    长度估算（words→tokens 启发式）：
        1 word ≈ 1.3 tokens（英文叙事文本经验值）。
        target_tokens 为近似目标，实际长度以生成的 word count 为准。
        调用方传入 ~512 / ~1024 / ~4096 均可。

    参数：
        n:                每个 split 期望的上下文数量
        split:            "train" | "validation" | "test"
        seed:             确定性种子
        queries_per_context: 每上下文的 query 数（2~4，默认 2）
        target_tokens:    近似目标 token 数（默认 1024 ≈ ~770 词）
    """
    import random as _rnd_mod

    rng = _rnd_mod.Random(seed + hash("needle_longctx") % 100000)

    # ── 1. 确定需要生成的总上下文数 ──
    # _split_train_val_test 用80/10/10切分；
    # 为确保目标 split 至少有 n 个上下文，生成足够的总数。
    if split == "train":
        n_total = max(-(-n * 10 // 8), 10)  # ceil(n / 0.8)
    else:
        n_total = max(n * 10, 20)  # 10% fraction → 10x

    target_words = max(1, int(target_tokens / 1.3))

    # ── 2. 生成所有上下文 ──
    contexts: list[tuple[str, str, list[tuple[str, int]]]] = []
    for ci in range(n_total):
        cid = f"nc-{seed}-{ci}"
        # 确定 needle 数量（1~3，由 rng 决定）
        n_needles = rng.randint(1, 3)
        needle_keys = rng.sample(_KEYS, n_needles)
        needle_vals = rng.sample(_VALUES, n_needles)
        needles = list(zip(needle_keys, needle_vals))

        # 填充文本
        filler = _build_filler(target_words, ci)

        # 把 needle facts 均匀嵌入 filler 中
        sentences = filler.split(". ")
        insert_positions = [
            max(1, int((i + 1) * len(sentences) / (n_needles + 1)))
            for i in range(n_needles)
        ]
        for pos, (k, v) in zip(insert_positions, needles):
            needle_sent = f"The magic number for {k} is {v}."
            sentences.insert(pos, needle_sent)
        text = ". ".join(sentences)
        if not text.endswith("."):
            text += "."

        contexts.append((cid, text, needles))

    # ── 3. context-level split（§16 §17）──
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    split_map: dict[str, str] = {}
    for ci in range(n_total):
        if ci < n_train:
            split_map[f"nc-{seed}-{ci}"] = "train"
        elif ci < n_train + n_val:
            split_map[f"nc-{seed}-{ci}"] = "validation"
        else:
            split_map[f"nc-{seed}-{ci}"] = "test"

    # ── 4. 筛选目标 split，取前 n 个上下文 ──
    target_contexts = [
        (cid, text, needles)
        for cid, text, needles in contexts
        if split_map[cid] == split
    ][:n]

    # ── 5. 为每个上下文生成 queries ──
    rows: list[Sample] = []
    for ci, (cid, text, needles) in enumerate(target_contexts):
        for qi in range(queries_per_context):
            needle = needles[qi % len(needles)]
            key, val = needle
            query = f"What is the magic number for {key}?"
            sample_id = f"nlx-{cid}-q{qi}"
            rows.append(
                Sample(
                    sample_id=sample_id,
                    context=text,
                    query=query,
                    answer=str(val),
                    split=split,
                    context_id=cid,
                )
            )

    # ── 6. 互斥性校验（§17 上下文级分层保证）──
    assert_xlevel_disjoint(rows)
    return rows


@_register("fineweb_edu")
def load_fineweb_edu(
    n: int,
    split: str = "train",
    seed: int = 0,
    *,
    target_tokens: int = 1024,
) -> list[Sample]:
    """Deterministic FineWeb-Edu-style long passages (V2 calibration regime).

    The public FineWeb-Edu corpus is not reachable offline here, so this loader
    assembles deterministic *educational-web-style* passages of roughly
    ``target_tokens`` tokens from a built-in paragraph pool. It is used only as
    a calibration context distribution aligned with the reference Heo et al.
    regime (long web text), not as an evaluation set.
    """
    target_words = max(1, int(target_tokens / 1.3))
    out: list[Sample] = []
    for ci in range(n):
        parts: list[str] = []
        wc = 0
        pidx = (ci * 7 + seed) % len(_EDU_PARAGRAPHS)
        while wc < target_words:
            para = _EDU_PARAGRAPHS[pidx % len(_EDU_PARAGRAPHS)]
            parts.append(para)
            wc += len(para.split())
            pidx += 1
        text = " ".join(parts)
        # deterministic light truncation at a sentence boundary
        words = text.split()
        if len(words) > target_words:
            cut = " ".join(words[:target_words])
            j = cut.rfind(".")
            text = cut[: j + 1] if j > len(cut) // 2 else cut + "."
        out.append(
            Sample(
                sample_id=f"fwe-{seed}-{ci}",
                context=text,
                query="",
                answer=None,
                split=split,
                context_id=f"fwe-{seed}-{ci}",
            )
        )
    return out


_EDU_PARAGRAPHS: list[str] = [
    "Photosynthesis is the process by which green plants, algae, and some "
    "bacteria convert light energy into chemical energy stored in glucose. "
    "Inside the chloroplasts, chlorophyll absorbs red and blue light while "
    "reflecting green light, which is why most leaves appear green. The light "
    "reactions split water molecules to release oxygen and produce ATP and "
    "NADPH, which then drive the Calvin cycle to fix carbon dioxide into sugar.",
    "The water cycle describes the continuous movement of water on, above, and "
    "below the Earth's surface. Evaporation turns liquid water into vapor, "
    "transpiration releases vapor from plant leaves, condensation forms clouds, "
    "and precipitation returns water to the ground. Groundwater and runoff "
    "eventually carry the water back to oceans and lakes, completing the loop.",
    "The American Civil War, fought from 1861 to 1865, arose from long-standing "
    "disputes over slavery, states' rights, and westward expansion. Eleven "
    "southern states seceded to form the Confederacy, while the Union retained "
    "the border states. Key battles at Antietam, Gettysburg, and Vicksburg "
    "shifted momentum, and the Emancipation Proclamation reframed the conflict "
    "around freedom before the Union prevailed in 1865.",
    "Fractions represent parts of a whole and can be added, subtracted, "
    "multiplied, and divided using consistent rules. To add fractions with "
    "different denominators, first find a common denominator, rewrite each "
    "fraction equivalently, and then combine the numerators. Multiplication is "
    "simpler: multiply numerators together and denominators together, then "
    "simplify the result by dividing by the greatest common factor.",
    "Earth's atmosphere is divided into layers distinguished by temperature "
    "gradients. The troposphere holds most weather and water vapor; the "
    "stratosphere contains the ozone layer that absorbs harmful ultraviolet "
    "radiation; the mesosphere is where most meteors burn up; and the "
    "thermosphere and exosphere gradually merge with space, hosting auroras "
    "and satellite orbits.",
    "The Industrial Revolution began in Britain in the late eighteenth century "
    "and transformed manufacturing through mechanization, steam power, and the "
    "factory system. Textile production was an early driver, followed by iron, "
    "coal, and rail. Urbanization accelerated as workers moved to cities, while "
    "new social questions about labor, education, and public health emerged.",
    "Algebra uses symbols to represent quantities and relationships. A linear "
    "equation such as three x plus five equals twenty can be solved by "
    "isolating the variable through inverse operations. Systems of equations "
    "are solved by substitution or elimination, and their solutions correspond "
    "to intersection points of the corresponding lines in the coordinate plane.",
    "The human digestive system breaks food into nutrients the body can absorb. "
    "Mechanical and chemical digestion begin in the mouth, continue in the "
    "stomach, and finish in the small intestine, where villi increase surface "
    "area. The large intestine reclaims water, and the liver and pancreas "
    "contribute bile and enzymes that support the breakdown of fats and proteins.",
    "Newton's three laws of motion describe how forces affect objects. An "
    "object at rest stays at rest unless acted on by a net force; acceleration "
    "is proportional to net force and inversely proportional to mass; and every "
    "action has an equal and opposite reaction. These laws underpin classical "
    "mechanics and explain phenomena from collisions to planetary orbits.",
    "Ecosystems consist of living communities interacting with nonliving "
    "components such as soil, water, and climate. Energy flows from producers "
    "to consumers and decomposers, while nutrients cycle through biotic and "
    "abiotic reservoirs. Biodiversity supports resilience, and disturbances "
    "such as fire or drought can shift an ecosystem between alternative states.",
    "The Constitution of the United States establishes a federal system with "
    "separated powers and checks and balances. Article One creates Congress, "
    "Article Two the presidency, and Article Three the judiciary. Amendments, "
    "including the Bill of Rights, protect individual liberties and have been "
    "added over time to reflect changing national commitments.",
    "Geometry studies points, lines, angles, and shapes and their properties. "
    "Triangles obey the Pythagorean theorem for right angles, circles relate "
    "radius, circumference, and area, and transformations such as translation, "
    "rotation, and reflection preserve distance and angle. Proofs connect "
    "definitions and theorems through deductive reasoning.",
    "Weather differs from climate: weather describes short-term atmospheric "
    "conditions, while climate summarizes long-term patterns. Air masses, "
    "fronts, and pressure gradients drive storms and temperature changes. "
    "Oceans and ice influence climate by storing and redistributing heat, and "
    "greenhouse gases regulate how much energy escapes to space.",
    "Ancient civilizations along major rivers developed agriculture, writing, "
    "and centralized government. Mesopotamian city-states invented cuneiform, "
    "Egyptian society organized around the Nile's floods, the Indus valley "
    "built planned cities, and early Chinese dynasties unified the Yellow "
    "River plain. Trade and migration spread ideas, technologies, and religions.",
]



@_register("needle_mcqa")
def load_needle_mcqa(
    n: int,
    split: str = "test",
    seed: int = 0,
    *,
    target_tokens: int = 1024,
) -> list[Sample]:
    """Long-context needle-in-a-haystack as a 4-choice task (§17 / P2-5).

    Converts the offline needle generator into a 4-choice item so the standard
    audit scoring path applies. Contexts are generated by a per-index
    deterministic RNG and partitioned by ``index % 10`` (test=1, validation=2,
    else train), so the first ``n`` items of a split are a stable prefix: the
    provider can request increasing ``n`` and still see the same item ``i``.
    """
    import random as _rnd_mod

    def _split_of(ci: int) -> str:
        if ci % 10 == 1:
            return "test"
        if ci % 10 == 2:
            return "validation"
        return "train"

    def _gen(ci: int) -> tuple[str, str, list[tuple[str, int]]]:
        rng = _rnd_mod.Random(seed * 1_000_003 + ci)
        n_needles = rng.randint(1, 3)
        needle_keys = rng.sample(_KEYS, n_needles)
        needle_vals = rng.sample(_VALUES, n_needles)
        needles = list(zip(needle_keys, needle_vals))
        target_words = max(1, int(target_tokens / 1.3))
        filler = _build_filler(target_words, ci)
        sentences = filler.split(". ")
        insert_positions = [
            max(1, int((i + 1) * len(sentences) / (n_needles + 1)))
            for i in range(n_needles)
        ]
        for pos, (k, v) in zip(insert_positions, needles):
            sentences.insert(pos, f"The magic number for {k} is {v}.")
        text = ". ".join(sentences)
        if not text.endswith("."):
            text += "."
        return f"nc-{seed}-{ci}", text, needles

    rng = _rnd_mod.Random(seed + 7919)
    out: list[Sample] = []
    ci = 0
    while len(out) < n:
        if _split_of(ci) != split:
            ci += 1
            continue
        cid, text, needles = _gen(ci)
        key, correct = needles[0]
        distractors = [v for v in _VALUES if v != correct]
        choices = [correct] + rng.sample(distractors, 3)
        rng.shuffle(choices)
        letter = "ABCD"[choices.index(correct)]
        query = "Choose the correct answer: " + " | ".join(
            f"{L}={v}" for L, v in zip("ABCD", choices)
        )
        out.append(
            Sample(
                sample_id=f"nlm-{cid}",
                context=text,
                query=query,
                answer=letter,
                split=split,
                context_id=cid,
            )
        )
        ci += 1
    assert_xlevel_disjoint(out)
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
    target_tokens: int | None = None,
) -> list[Sample]:
    """按 dataset 名取确定性 split 的 n 个样本。

    这层不让 runner 感知 HF / synthetic 的差异——runner 只看到 Stream of Sample。
    target_tokens：仅 needle 类生成器使用（长上下文长度）。
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown dataset {name!r}. Available: {sorted(_REGISTRY.keys())}"
        )
    kwargs: dict[str, Any] = {}
    if target_tokens is not None:
        kwargs["target_tokens"] = int(target_tokens)
    rows = _REGISTRY[name](n, split=split, seed=seed, **kwargs)
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
    "load_needle_longctx",
    "load_needle_mcqa",
    "load_fineweb_edu",
]
