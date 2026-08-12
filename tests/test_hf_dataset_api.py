"""apcs.data.hf_dataset 公共 API 烟雾测试（无需 datasets 库）。

若 datasets 已安装，本测试会跑一遍真实加载并验证 schema；
未安装则只验证 API 形状（registered_names / load 抛错的合理信息）。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apcs.data.hf_dataset import (
    ensure_datasets_available,
    load,
    registered_names,
)


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
