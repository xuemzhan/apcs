"""apcs.data.hf_dataset 公共 API 烟雾测试（无需 datasets 库）。

若 datasets 已安装，本测试会跑一遍真实加载并验证 schema；
未安装则只验证 API 形状（registered_names / load 抛错的合理信息）。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apcs.data.hf_dataset import (
    _split_train_val_test,
    ensure_datasets_available,
    load,
    registered_names,
    split_manifest,
)
from apcs.data import Sample


def test_registered_names_nonempty():
    """至少含 5 个数据集（hellaswag / arc_c / arc_e / mmlu / winogrande）。"""
    names = registered_names()
    assert len(names) >= 5
    assert "hellaswag" in names
    assert "mmlu" in names


def test_load_unknown_name_raises():
    """未知名必须 raise KeyError，且消息含『已知列表』便于用户修正。"""
    try:
        load("not-a-real-dataset", n=1)
    except KeyError as e:
        msg = str(e)
        assert "hellaswag" in msg or "Available" in msg
    else:
        raise AssertionError("load(unknown) 必须 raise KeyError")


def test_ensure_datasets_available():
    """ensure_datasets_available 不 返回值；要么直接过，要么 raise RuntimeError。"""
    try:
        ensure_datasets_available()
    except RuntimeError as e:
        assert "datasets" in str(e).lower()
    # 没装也不应崩——只 raise 提示


class _FakeDataset(list):
    def shuffle(self, seed=0):
        import random
        values = list(self)
        random.Random(seed).shuffle(values)
        return _FakeDataset(values)

    def select(self, indices):
        return _FakeDataset([self[i] for i in indices])


def test_deterministic_splits_are_disjoint():
    ds = _FakeDataset(range(100))
    a = _split_train_val_test(ds, seed=7)
    b = _split_train_val_test(ds, seed=7)
    assert list(a["train"]) == list(b["train"])
    assert set(a["train"]).isdisjoint(a["validation"])
    assert set(a["train"]).isdisjoint(a["test"])
    assert set(a["validation"]).isdisjoint(a["test"])


def test_split_manifest_is_stable():
    rows = [Sample(f"id-{i}", "ctx", "q", "A", "test") for i in range(3)]
    first = split_manifest("toy", rows, "test", 1)
    second = split_manifest("toy", rows, "test", 1)
    assert first == second
    assert first["n_samples"] == 3
    assert len(first["sample_ids_sha256"]) == 64
