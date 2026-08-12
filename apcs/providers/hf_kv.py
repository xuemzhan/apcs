"""HF KVProvider —— 真实 HuggingFace 路径（TODO(real-gpu)，未实现）。

本模块存在不是为了"提供 HF 功能"，而是为了：

    1. 在 cfg 设 `provider.kv: hf` 时显式 raise，
       让"真实路径不可用"是**错误栈可见**的失败信号，
       而不是悄悄回退到合成（§75 诚实性）。
    2. 给将来的实现者一个**协议起点**：见 `KVProvider` 协议
       （apcs/providers/__init__.py）的方法签名 + 当前 docstring 列出的
       关键设计点。

**实现指南（接入真实 GPU 路径时按本节填充）**：

    - 在 `pyproject.toml` 的 `[project.optional-dependencies].gpu` 已经
      列出 torch + transformers + datasets；本模块顶层 import 时把这些
      包作为**可选**依赖（try/except ImportError → raise RuntimeError）。
    - `open(cfg)`：加载 Teacher / Student 两个 HF 模型（cfg.teacher.model_id,
      cfg.student.model_id），分别保存到 self._teacher, self._student；
      tokenizer 共享一个；dtype 用 cfg.teacher.dtype。
    - `iter_calibration(n_samples, seed)`：取 cfg.datasets.fidelity / 校准子集，
      逐样本 forward 抓 past_key_values。
      关键：必须把 K 与 V 同时吐出（per-shot 校准需要 K+V 联合形式）。
    - `iter_eval(n_samples, seed)`：与 iter_calibration 共享同一 dataset，
      但**严格从 test split 取**（设计稿 §16 / §36 / §70；切勿跑到 train
      集）。held-out 语义由 "split=test" 过滤器保证。
    - `close()`：del + torch.cuda.empty_cache()。
    - `describe()`：返回 model_id / commit / dtype / device_map /
      tokenizer / context_length / dataset / split，可直接对接 §64 metadata。

**为什么不能简单 `from_pretrained` 然后返回**：见 §75 —— 真实路径必须：
    (a) 暴露 teacher 还是 student 接到了 KV 注入（注入一致性）；
    (b) 把 torch tensor 序列化为 np.ndarray 喂给 runner
        （uniform interface；避免 runner 出现 if torch: ... else numpy: ...）；
    (c) 错误时显式 raise（不允许 silent re-prefill，§52.8）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

from . import CalibrationSample


@dataclass
class HFKVProvider:
    """TODO(real-gpu)：HF 真实 KVProvider 骨架。

    任何字段/方法都先 raise，让上层在 open() 阶段就崩溃；
    避免「构造出半成品、等 runner 调用时才发现没接好」的延迟失败。
    """
    kind: str = "hf"

    def open(self, cfg: dict[str, Any]) -> None:
        raise NotImplementedError(
            "HFKVProvider 尚未实现（TODO(real-gpu)）。"
            "接入真实 GPU 路径请按 apcs/providers/hf_kv.py 模块 docstring 的"
            "实现指南填充本类。当前 cfg `provider.kv: hf` 必然失败，"
            "不要为了跑通而临时设成 synthetic —— 设计稿 §75 禁止。"
        )

    def iter_calibration(
        self, n_samples: int, seed: int
    ) -> Iterator[CalibrationSample]:
        raise NotImplementedError(self.open.__doc__)

    def iter_eval(
        self, n_samples: int, seed: int
    ) -> Iterator[CalibrationSample]:
        raise NotImplementedError(self.open.__doc__)

    def close(self) -> None:
        raise NotImplementedError(self.open.__doc__)

    def describe(self) -> dict[str, Any]:
        raise NotImplementedError(self.open.__doc__)


__all__ = ["HFKVProvider"]
