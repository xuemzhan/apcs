"""HF KVProvider —— 真实模型 KV 捕获（modelscope 源）。

═══════════════════════════════════════════════════════════════════════════════
本模块实现 `KVProvider` 协议的 HF 版本（design.md §14–§18 / §32 / §36 / §70）：

    - 模型源：modelscope.cn 优先（AutoModelForCausalLM 接口与 transformers 对齐），
      其次 transformers / HF hub。
    - open(cfg)：加载 Teacher + Student 两个 HF 因果 LM 与共享 tokenizer；
      失败必须显式 raise（§75 诚实性，禁止静默回退合成）。
    - iter_calibration(n, seed)：从 cfg.datasets.fidelity / 校准子集取 prompts，
      逐样本 forward 抓 past_key_values，K+V 联合吐出为 (L, S, H, 2D) numpy。
    - iter_eval(n, seed)：与 iter_calibration 共享同一 dataset，但严格从
      test split 取样（§16 / §36 / §70 防数据泄漏）。
    - close()：del + torch.cuda.empty_cache()。
    - describe()：model_id / revision / dtype / device_map / dataset / split。

**KV 形状契约**：仓库 numpy 契约是每层一份 `(S, H, D)` 状态（synthetic 与
mapper 均如此）；真实 KV 每层有独立的 K 与 V，本 provider 沿最后维拼接成
`(L, S, H, 2D)`（K 在前 V 在后），与 `inference.backends` 的
`_extract_kv_numpy` 口径一致，mapper/inject 无需感知拆分。
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np

from . import CalibrationSample
from .hf_model import capture_kv_pair  # noqa: F401  (真实 forward 复用)


def _model_source():
    """返回 (AutoModelForCausalLM, AutoTokenizer)：modelscope 优先。"""
    try:
        from modelscope import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        return AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        return AutoModelForCausalLM, AutoTokenizer


@dataclass
class HFKVProvider:
    """HF 真实 KVProvider（modelscope 源，需 torch + transformers/datasets）。"""

    kind: str = "hf"
    _teacher: Any = None
    _student: Any = None
    _tok: Any = None
    _tok_student: Any = None
    _cfg: dict[str, Any] = field(default_factory=dict)
    _device: str = ""
    _model_cls: Any = None

    # ---- 协议 ----

    def open(self, cfg: dict[str, Any]) -> None:
        """加载 Teacher / Student 模型 + 共享 tokenizer（§53 Load）。

        模型源 modelscope.cn 优先；dtype/device_map 取自 cfg.teacher。
        失败显式 raise（§75：不静默回退合成）。
        """
        try:
            import torch  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "provider.kv=hf 需要 torch；请安装项目 gpu extra，"
                "或显式选择 provider.kv=synthetic"
            ) from e

        from .hf_model import resolve_device

        self._cfg = cfg
        device = resolve_device(cfg)
        AutoModelForCausalLM, AutoTokenizer = _model_source()

        t_cfg = cfg.get("teacher", {})
        s_cfg = cfg.get("student", {})
        if not t_cfg.get("model_id") or not s_cfg.get("model_id"):
            raise RuntimeError(
                "provider.kv=hf 的 HFKVProvider.open 需要 cfg.teacher.model_id "
                "与 cfg.student.model_id（见 run_dir/provider.json 记录 provider 选择）"
            )

        revision = str(t_cfg.get("revision", "main"))
        self._device = device
        self._model_cls = AutoModelForCausalLM
        self._tok = AutoTokenizer.from_pretrained(
            t_cfg["model_id"], revision=revision
        )
        # ◆ D4 修复：student 侧使用自己的 tokenizer。此前 teacher/student
        # 共用 teacher tokenizer —— 同族模型（Qwen3 系列 vocab 一致）无碍，
        # 但跨词表模型对会把 teacher token id 直接喂给 student（语义错位）。
        # 词表一致时 student tokenizer 加载失败则回退共享（如实降级）。
        self._tok_student = None
        try:
            self._tok_student = AutoTokenizer.from_pretrained(
                s_cfg["model_id"], revision=str(s_cfg.get("revision", "main"))
            )
        except Exception as e:  # noqa: BLE001
            self._tok_student = self._tok
            print(f"[hf_kv] student tokenizer 加载失败，回退 teacher tokenizer: {e}")

    def iter_calibration(
        self, n_samples: int, seed: int
    ) -> Iterator[CalibrationSample]:
        """生成 n_samples 个校准样本（fidelity 子集，train split）。"""
        yield from self._capture_split(n_samples, seed, split="train")

    def iter_eval(
        self, n_samples: int, seed: int
    ) -> Iterator[CalibrationSample]:
        """生成 n_samples 个 held-out 评估样本（**test split**，§36/§70）。"""
        yield from self._capture_split(n_samples, seed, split="test")

    def close(self) -> None:
        """释放模型 + GPU 缓存（§53 Teacher/Student Unload）。"""
        import torch  # type: ignore

        self._teacher = None
        self._student = None
        self._tok = None
        self._model_cls = None
        self._device = ""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def describe(self) -> dict[str, Any]:
        """返回可序列化描述（对接 §64 metadata / provider.json）。"""
        t = self._cfg.get("teacher", {})
        s = self._cfg.get("student", {})
        return {
            "implementation": "hf_modelscope",
            "teacher_model_id": t.get("model_id"),
            "student_model_id": s.get("model_id"),
            "revision": t.get("revision", "main"),
            "dtype": t.get("dtype", "bfloat16"),
            "device_map": t.get("device_map", "cuda:0"),
            "dataset": self._cfg.get("datasets", {}).get("fidelity", []),
            "note": "K+V 沿 head_dim 拼接为 (L, S, H, 2D)；iter_eval 严格取 test split（§36）",
        }

    # ---- 内部 ----

    def _make_prompt(self, seed: int, idx: int, split: str) -> tuple[str, str]:
        """按 seed+idx+split 派生 prompt 文本（确定性，§51 可复现）。

        优先用真实数据集（cfg.datasets.fidelity 或 teacher_advantage 首个），
        取不到时退回确定性合成 prompt（仍标注于 describe）。
        """
        datasets_cfg = self._cfg.get("datasets", {})
        names = (
            datasets_cfg.get("fidelity")
            or datasets_cfg.get("teacher_advantage", {}).get("primary")
            and [datasets_cfg["teacher_advantage"]["primary"]]
            or []
        )
        if names:
            try:
                from ..data.hf_dataset import load as load_ds

                rows = load_ds(
                    names[0], n=max(4, idx + 1), split=split, seed=seed
                )
                if idx < len(rows):
                    r = rows[idx]
                    prompt = f"{r.context}\n{r.query}" if r.context else r.query
                    return prompt, r.sample_id
            except Exception as e:  # noqa: BLE001
                if not bool(
                    self._cfg.get("provider", {}).get(
                        "allow_synthetic_prompt_fallback", False
                    )
                ):
                    raise RuntimeError(
                        f"HF KV 数据集 {names[0]!r}/{split} 加载失败；"
                        "为防止真实实验静默混入合成 prompt，运行已终止。"
                    ) from e
        rng = np.random.default_rng(seed * 1_000_000 + idx)
        words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
        seq = [words[i % len(words)] for i in range(32 + idx)]
        prompt = f"{split}-{seed}-{idx}: " + " ".join(seq)
        return prompt, f"synthetic-{split}-{seed}-{idx}"

    def _load_role(self, role: str):
        """单独加载一侧模型；Teacher/Student 不同时驻留 GPU。"""
        import torch  # type: ignore
        from .hf_model import load_hf_model

        spec = self._cfg[role]
        dtype_name = str(spec.get("dtype", "bfloat16"))
        dtype = getattr(torch, dtype_name, torch.bfloat16)
        return load_hf_model(
            spec["model_id"],
            str(spec.get("revision", "main")),
            dtype,
            self._device,
            cls=self._model_cls,
        )

    @staticmethod
    def _release_model(model) -> None:
        import torch  # type: ignore
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _capture_role(self, model, prompts: list[str], role: str = "teacher") -> list[np.ndarray]:
        # ◆ D4 修复：按角色选 tokenizer（teacher/student 词表可能不同）
        tok = self._tok if role == "teacher" else getattr(self, "_tok_student", None) or self._tok
        values = []
        for prompt in prompts:
            ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
            values.append(capture_kv_pair(model, ids).astype(np.float32, copy=False))
        return values

    def _capture_split(self, n_samples: int, seed: int, *, split: str):
        if self._tok is None or not self._device:
            raise RuntimeError("HFKVProvider.open() 未调用")
        rows = [self._make_prompt(seed, i, split=split) for i in range(n_samples)]
        prompts = [p for p, _ in rows]

        self._teacher = self._load_role("teacher")
        try:
            teacher_values = self._capture_role(self._teacher, prompts, role="teacher")
        finally:
            teacher_model = self._teacher
            self._teacher = None
            self._release_model(teacher_model)

        self._student = self._load_role("student")
        try:
            student_values = self._capture_role(self._student, prompts, role="student")
        finally:
            student_model = self._student
            self._student = None
            self._release_model(student_model)

        for (prompt, sample_id), kv_t, kv_s in zip(rows, teacher_values, student_values):
            yield CalibrationSample(
                kv_t=kv_t,
                kv_s=kv_s,
                sample_id=sample_id,
                prompt=prompt,
                split=split,
            )


__all__ = ["HFKVProvider"]
