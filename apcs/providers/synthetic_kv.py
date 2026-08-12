"""Synthetic KVProvider —— 当前实现的仿真数据源。

这是离线 demo 的唯一已实现 provider；与 mapper/runner.py 内部原先的
`_synth_calibration_set` 共享同一套「共享 W_t / W_s + 独立 latent Z」语义，
§32 bug-3 修复要点已继承。

**对外接口**遵循 apcs.providers.KVProvider 协议 —— runner 只跟协议打交道，
因此将来接入 HF 模型只需新增一个实现类，主 runner 不动。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np

from . import CalibrationSample


@dataclass
class SyntheticKVProvider:
    """合成 KVProvider（§32 calibration 语义 + held-out seed 隔离）。

    关键不变量（设计.md §32 bug-3 修复）：
        calibration 的 W_t/W_s 由 `seed` 派生一次，后续样本复用；
        同一 (n_t, n_s, head_dim, H) 下，**改 seed 生成全新一组** W_t/W_s，
        从而 calibration 子空间与 held-out eval 子空间联合不交。
    """
    seed: int = 0
    kind: str = "synthetic"
    _rng: np.random.Generator | None = None
    _n_t: int = 0
    _n_s: int = 0
    _n_kv: int = 0
    _head_dim: int = 0
    _W_t: list[np.ndarray] | None = None
    _W_s: list[np.ndarray] | None = None

    # ---- 协议 ----

    def open(self, cfg: dict[str, Any]) -> None:
        """从 cfg 读取模型/序列维度。"""
        teacher = cfg["teacher"]
        student = cfg["student"]
        self._n_t = int(teacher["num_layers"])
        self._n_s = int(student["num_layers"])
        self._n_kv = int(teacher["num_kv_heads"])
        self._head_dim = int(teacher["head_dim"])
        # 共享 W_t / W_s —— 一次生成，跨样本复用（§32 校准语义）
        rng = np.random.default_rng(self.seed)
        self._W_t = [
            rng.standard_normal((self._head_dim, self._head_dim)) / np.sqrt(self._head_dim)
            for _ in range(self._n_t)
        ]
        self._W_s = [
            rng.standard_normal((self._head_dim, self._head_dim)) / np.sqrt(self._head_dim)
            for _ in range(self._n_s)
        ]
        self._rng = rng

    def iter_calibration(
        self, n_samples: int, seed: int
    ) -> Iterator[CalibrationSample]:
        """生成 n_samples 个校准样本（共享 W_t/W_s + 独立 latent Z）。"""
        assert self._W_t is not None and self._W_s is not None, "open() 未调用"
        for i in range(n_samples):
            # 校准样本的 latent 种子在「calibration 子空间」内
            latent_seed = self.seed * 1_000_000 + i
            yield self._make_sample(latent_seed, scope="calib", idx=i)

    def iter_eval(
        self, n_samples: int, seed: int
    ) -> Iterator[CalibrationSample]:
        """生成 n_samples 个 held-out 评估样本（evaluate-only）。

        关键：与 calibration 的 (seed, latent_seed) 联合不交——
        与同种子下 `mapper/runner.py::_synth_calibration_set` 行为一致。
        评估层的样本种子 = `seed * 1_000_000 + 1_000_000 + i`：
        偏移 1_000_000 把 eval 推到一个**完全不重叠**的子空间，
        防数据泄漏（de-collide）—— 测试 test_held_out_eval_samples_do_not_collide_with_calib
        已锁定。
        """
        assert self._W_t is not None and self._W_s is not None, "open() 未调用"
        for i in range(n_samples):
            latent_seed = self.seed * 1_000_000 + 1_000_000 + i
            yield self._make_sample(latent_seed, scope="eval", idx=i)

    def close(self) -> None:
        """释放资源（无 GPU 占用，置空即可）。"""
        self._W_t = None
        self._W_s = None
        self._rng = None

    def describe(self) -> dict[str, Any]:
        return {
            "implementation": "synthetic_latent",
            "note": (
                "合成数据：共享 W_t/W_s + 独立 latent Z，"
                "calib 与 held-out eval 通过 latent_seed 偏移 1_000_000 隔离。"
            ),
            "n_t": self._n_t,
            "n_s": self._n_s,
            "n_kv_heads": self._n_kv,
            "head_dim": self._head_dim,
            "w_seed": self.seed,
        }

    # ---- 内部 ----

    def _make_sample(
        self, latent_seed: int, *, scope: str, idx: int
    ) -> CalibrationSample:
        """生成单个 (kv_t, kv_s) 样本（H 复用 W_t/W_s，latent Z 独立）。"""
        seq_len = 128  # 与现有 _synth_calibration_set 默认行为一致
        rng = np.random.default_rng(latent_seed)
        Z = rng.standard_normal((seq_len, self._n_kv, self._head_dim))
        kv_t = np.zeros((self._n_t, seq_len, self._n_kv, self._head_dim), dtype=np.float32)
        kv_s = np.zeros((self._n_s, seq_len, self._n_kv, self._head_dim), dtype=np.float32)
        for l in range(self._n_t):
            kv_t[l] = Z @ self._W_t[l]
        for l in range(self._n_s):
            kv_s[l] = Z @ self._W_s[l]
        return CalibrationSample(
            kv_t=kv_t,
            kv_s=kv_s,
            sample_id=f"{scope}-{idx}",
        )


__all__ = ["SyntheticKVProvider"]
