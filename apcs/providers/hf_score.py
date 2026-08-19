"""HF ScoreProvider —— §37 7 方法的真实 LLM 评分（modelscope 源）。

═══════════════════════════════════════════════════════════════════════════════
实现 `ScoreProvider` 协议的 HF 版本：

    - score(method, sample_id, seed)：用加载的模型对 prompt 做多选/续写评分。
      method ∈ {student, teacher, text, ridge, base_only, base_plus_adv,
      full_apcs}（§37 报口径）：
        * student / teacher：对应模型的直接零样本推理得分；
        * text / ridge / base_only / base_plus_adv / full_apcs：需要先做
          KV 注入（handoff），再以注入后的 Student 缓存评分 —— 由外部
          以 run_id / cfg 提供注入路径，本 provider 只负责"用哪个模型打分"。
          未注入的兜底 = 该模型的普通得分（并标注），确保可跑。
    - decision(method, sample_id, seed)：同一模型输出 argmax 选项 ID（§45 JCR）。

**形状/语义**：得分 = 模型 logits 计算 4 选项正确率（0–1），比 synthetic
更接近 §15 Scoring 的真实定义。GPU 不可计算时 open() 显式 raise（§75）。
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .hf_model import load_hf_model, resolve_device

_METHODS = {
    "student",
    "teacher",
    "text",
    "ridge",
    "base_only",
    "base_plus_adv",
    "full_apcs",
}


def _model_source():
    try:
        from modelscope import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        return AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        return AutoModelForCausalLM, AutoTokenizer


@dataclass
class HFScoreProvider:
    """HF 真实 ScoreProvider（modelscope 源）。"""

    kind: str = "hf"
    _model: Any = None
    _tok: Any = None
    _cfg: dict[str, Any] = field(default_factory=dict)

    def open(self, cfg: dict[str, Any]) -> None:
        """加载评分模型（student 侧）。失败显式 raise。"""
        import torch  # type: ignore

        self._cfg = cfg
        device = resolve_device(cfg)
        AutoModelForCausalLM, AutoTokenizer = _model_source()
        s_cfg = cfg.get("student", {})
        if not s_cfg.get("model_id"):
            raise RuntimeError("HFScoreProvider.open 需要 cfg.student.model_id")
        dtype_name = str(s_cfg.get("dtype", "bfloat16"))
        dtype = getattr(torch, dtype_name, torch.bfloat16)
        revision = str(s_cfg.get("revision", "main"))
        self._model = load_hf_model(
            s_cfg["model_id"], revision, dtype, device, cls=AutoModelForCausalLM
        )
        self._tok = AutoTokenizer.from_pretrained(
            s_cfg["model_id"], revision=revision
        )
        self._model.eval()

    def score(self, method: str, sample_id: str, seed: int) -> float:
        """`method` 在 sample_id 上的得分（0–1 正确率）。

        对 teacher 方法用 teacher 侧模型打分（若无 teacher 模型则复用
        student 模型并标注）；对注入类方法（ridge/base_only/base_plus_adv/
        full_apcs）以 student 模型零样本打分（未注入的兜底，见模块 docstring）。
        """
        if method not in _METHODS:
            raise KeyError(f"Unknown method {method!r}")
        raise NotImplementedError(
            "HFScoreProvider 尚未接入带正确答案的 Sample 与各 method 的注入 cache；"
            "为避免把 Student 零样本概率误报为真实 CHG，当前显式阻断。"
        )

    def decision(self, method: str, sample_id: str, seed: int) -> int:
        """`method` 在 sample_id 上的决策 ID（0/1 伯努利，§45 JCR）。"""
        if method not in _METHODS:
            raise KeyError(f"Unknown method {method!r}")
        raise NotImplementedError(
            "HFScoreProvider decision 尚未接入真实候选答案与 handoff cache。"
        )

    def close(self) -> None:
        import torch  # type: ignore

        self._model = None
        self._tok = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def describe(self) -> dict[str, Any]:
        s = self._cfg.get("student", {})
        return {
            "implementation": "hf_modelscope_score",
            "evidence_grade": "blocked_until_sample_and_cache_wiring",
            "model_id": s.get("model_id"),
            "revision": s.get("revision", "main"),
            "dtype": s.get("dtype", "bfloat16"),
            "note": (
                "真实 LLM 评分：对 prompt 计算 4 选项 logits 正确率。"
                "注入类方法以 student 零样本兜底（未接线 KV 注入），"
                "接入 run_id 注入路径后即为真值。"
            ),
        }

    # ---- 内部 ----

    def _make_prompt(self, sample_id: str, seed: int) -> str:
        """派生确定性 prompt（§51 可复现），尽量用真实数据集。"""
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

                rows = load_ds(names[0], n=32)
                idx = int(np.argmax(np.frombuffer(np.asarray([sample_id], dtype="S32").tobytes(), dtype=np.uint8)))
                idx = idx % len(rows)
                r = rows[idx]
                return f"{r.context}\n{r.query}" if r.context else r.query
            except Exception:  # noqa: BLE001
                pass
        rng = np.random.default_rng(seed * 1_000_000 + len(sample_id))
        return f"{sample_id}: " + " ".join(
            rng.choice(["a", "b", "c", "d"], size=8).tolist()
        )

    def _score_choices(self, prompt: str) -> float:
        """对 4 选项评分：返回 argmax 选项 logits 的 softmax 概率（0–1）。"""
        import torch  # type: ignore

        choices = ["A", "B", "C", "D"]
        with torch.no_grad():
            # 简单做法：对 prompt 做一次 forward，取最后一个位置 logits 中
            # A/B/C/D 的 softmax 概率作为"正确率"（离线可跑，非评测集精确指标）
            enc = self._tok(prompt, return_tensors="pt").input_ids.to(self._model.device)
            logits = self._model(enc).logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)[0]
            token_ids = [self._tok.convert_tokens_to_ids(c) for c in choices]
            vals = [probs[t].item() if t != -1 else 0.0 for t in token_ids]
        mx = max(vals) if vals else 0.0
        return mx if mx > 0 else 0.5  # 兜底 0.5 避免全零（未接 tokenizer 时）

    def _open_if_needed(self, cfg: dict[str, Any]) -> None:
        if self._model is None:
            self.open(cfg)


__all__ = ["HFScoreProvider"]
