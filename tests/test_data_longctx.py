"""§17 needle_longctx 数据层测试（纯离线，无网络依赖）。

覆盖：
    (a) context_id ↔ split 互斥性
    (b) 同一 context_id 内 query 共享相同 context 文本
    (c) answer 与嵌入 needle facts 一致
    (d) 长度启发式单调性（target_tokens=512 < 4096）
    (e) assert_xlevel_disjoint 在泄漏行上正确报错
    (f) load_mmlu 在 Hub 不可用时拒绝合成假数据
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apcs.data import Sample, assert_xlevel_disjoint
from apcs.data.hf_dataset import load, load_needle_longctx


# -----------------------------------------------------------------------
# (a) needle loader: 每个 context_id 恰好映射到一个 split
# -----------------------------------------------------------------------

def test_needle_longctx_context_id_maps_to_one_split():
    """同一 context_id 的所有行必须属于同一 split。"""
    for s in ("train", "validation", "test"):
        rows = load_needle_longctx(n=4, split=s, seed=0)
        cid_splits: dict[str, set[str]] = {}
        for r in rows:
            cid = r.context_id or r.sample_id
            cid_splits.setdefault(cid, set()).add(r.split)
        for cid, splits in cid_splits.items():
            assert len(splits) == 1, (
                f"context_id={cid!r} spans splits {splits}"
            )
            assert splits.pop() == s


def test_needle_longctx_all_splits_collectively_disjoint():
    """跨 split 调用时，train/val/test 的 context_id 集合互斥。"""
    all_rows: list[Sample] = []
    for s in ("train", "validation", "test"):
        all_rows.extend(load_needle_longctx(n=4, split=s, seed=42))
    # 不应 raise
    assert_xlevel_disjoint(all_rows)


# -----------------------------------------------------------------------
# (b) 同一 context_id 内 2-4 queries 共享相同 context 文本
# -----------------------------------------------------------------------

def test_needle_longctx_queries_share_context_text():
    """同一 context_id 的所有 query 的 context 字段完全相同。"""
    for qpc in (2, 3, 4):
        rows = load_needle_longctx(
            n=6, split="train", seed=7, queries_per_context=qpc,
        )
        ctx_by_cid: dict[str, list[str]] = {}
        for r in rows:
            ctx_by_cid.setdefault(r.context_id or r.sample_id, []).append(r.context)
        for cid, texts in ctx_by_cid.items():
            assert len(texts) == qpc, f"cid={cid}: expected {qpc} rows, got {len(texts)}"
            assert len(set(texts)) == 1, f"cid={cid}: context texts differ"


# -----------------------------------------------------------------------
# (c) answer 与嵌入 needle facts 一致
# -----------------------------------------------------------------------

def test_needle_longctx_answers_match_needles():
    """query 提到的 key 对应的 answer 必须与嵌入的 needle fact 一致。"""
    rows = load_needle_longctx(n=10, split="test", seed=13, queries_per_context=3)
    # 从每行 query 提取 key，再从 context 提取该 key 对应的 value
    import re
    needle_re = re.compile(r"The magic number for (\w+) is (\d+)\.")
    for r in rows:
        key_match = re.search(r"What is the magic number for (\w+)\?", r.query)
        assert key_match, f"unexpected query format: {r.query!r}"
        key = key_match.group(1)
        # 在 context 中找该 key 的 needle fact
        facts = {m.group(1): m.group(2) for m in needle_re.finditer(r.context)}
        assert key in facts, f"needle for {key!r} not found in context"
        assert r.answer == facts[key], (
            f"answer mismatch: query asks {key!r}, context says "
            f"{facts[key]!r}, answer is {r.answer!r}"
        )


# -----------------------------------------------------------------------
# (d) 长度启发式单调性：512-target < 4096-target
# -----------------------------------------------------------------------

def test_needle_longctx_length_monotonicity():
    """target_tokens=512 的 context 应短于 target_tokens=4096 的 context。"""
    short = load_needle_longctx(
        n=4, split="train", seed=0, target_tokens=512,
    )
    long = load_needle_longctx(
        n=4, split="train", seed=0, target_tokens=4096,
    )
    avg_short = sum(len(r.context) for r in short) / max(len(short), 1)
    avg_long = sum(len(r.context) for r in long) / max(len(long), 1)
    assert avg_short < avg_long, (
        f"512-target ({avg_short:.0f} chars) should be shorter "
        f"than 4096-target ({avg_long:.0f} chars)"
    )


# -----------------------------------------------------------------------
# (e) assert_xlevel_disjoint 在泄漏行上 raise ValueError
# -----------------------------------------------------------------------

def test_assert_xlevel_disjoint_raises_on_leakage():
    """同一 context_id 出现在两个 split 时必须 raise ValueError。"""
    leaking = [
        Sample("s1", "ctx", "q", "a", "train", context_id="shared"),
        Sample("s2", "ctx", "q", "a", "test", context_id="shared"),
    ]
    try:
        assert_xlevel_disjoint(leaking)
    except ValueError as e:
        assert "shared" in str(e)
        assert "split" in str(e).lower() or "泄漏" in str(e)
    else:
        raise AssertionError(
            "assert_xlevel_disjoint should raise ValueError on cross-split context_id"
        )


def test_assert_xlevel_disjoint_passes_on_clean_rows():
    """不同 context_id 各在单一 split 时不应 raise。"""
    clean = [
        Sample("s1", "ctx", "q", "a", "train", context_id="c1"),
        Sample("s2", "ctx", "q", "a", "train", context_id="c2"),
        Sample("s3", "ctx", "q", "a", "test", context_id="c3"),
    ]
    assert_xlevel_disjoint(clean)  # 不应 raise


# -----------------------------------------------------------------------
# (f) load_mmlu 在 Hub 不可用时 raise RuntimeError（拒绝合成假数据）
# -----------------------------------------------------------------------

class _FakeDatasetsModule:
    """模拟 datasets 库：load_dataset 永远失败（Hub 不可达）。"""

    @staticmethod
    def load_dataset(*args, **kwargs):
        raise ConnectionError("Hub unreachable in test")


def test_load_mmlu_raises_on_hub_failure(monkeypatch):
    """cais/mmlu 和 hails/mmlu_no_train 都失败时 raise RuntimeError，不合成。"""
    monkeypatch.setattr(
        "apcs.data.hf_dataset._import_datasets",
        lambda: _FakeDatasetsModule,
    )
    try:
        load("mmlu", n=4, split="train", seed=0)
    except RuntimeError as e:
        msg = str(e).lower()
        assert "mmlu" in msg or "unavailable" in msg or "refusing" in msg
    else:
        raise AssertionError(
            "load_mmlu must raise RuntimeError when Hub unavailable"
        )


# -----------------------------------------------------------------------
# load() 注册表可发现 needle_longctx
# -----------------------------------------------------------------------

def test_needle_longctx_registered():
    """needle_longctx 通过 load() 注册表可访问。"""
    rows = load("needle_longctx", n=3, split="train", seed=99)
    assert len(rows) >= 1
    assert all(r.context_id is not None for r in rows)
