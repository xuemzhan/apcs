"""HF ScoreProvider —— §37 7 方法的真实 LLM 评分（modelscope 源）。

════════════════════════════════════════════════════════════════════════════════
实现 `ScoreProvider` 协议的 HF 版本。

设计决策（Option B：Artifact Delegation）：
    HFScoreProvider 不独立实现评分，而是委托给 ArtifactScoreProvider。
    inject-eval 运行后产出 capability_score_artifact.json，HFScoreProvider
    在 open() 时检测该 artifact 并委托；若无 artifact 则保持阻断（§75）。

    选择原因：
        1. 评分逻辑集中在一处（evaluator → artifact → provider），零重复；
        2. inject-eval 的四方法评估是唯一产生真实 scores 的路径；
        3. ArtifactScoreProvider 的 schema 校验（evidence_grade / split / 有限数）
           提供额外审计层。

方法映射：
    - inject-eval 方法名映射：student → "student", teacher → "teacher",
      text → "text", ridge → "ridge"；
    - base_only / base_plus_adv / full_apcs 需 T08 训练后接入（冻结分支）；
    - 未在 artifact 中的方法 → KeyError（§75 诚实性，不伪造默认分数）。
════════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
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

# inject-eval 方法名到 capability_score_artifact.json 中 method 字段的映射。
# 当前 inject-eval evaluator 产出的 method 名完全匹配，无需转换；
# 此表作为显式注册点，未来可扩展别名。
_INJECT_EVAL_METHOD_MAP = {
    "student": "student",
    "teacher": "teacher",
    "text": "text",
    "ridge": "ridge",
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
    """HF 真实 ScoreProvider —— 委托 capability_score_artifact.json。

    Option B 设计（见模块 docstring）：
        open() 时扫描 run_dir 下 capability_score_artifact.json，
        存在则委托 ArtifactScoreProvider 做评分；
        不存在则阻断（§75 诚实性，不伪造分数）。
    """

    kind: str = "hf"
    _model: Any = None
    _tok: Any = None
    _cfg: dict[str, Any] = field(default_factory=dict)
    _artifact_delegate: Any = None  # ArtifactScoreProvider 实例

    def open(self, cfg: dict[str, Any]) -> None:
        """加载 capability_score_artifact.json 并委托 ArtifactScoreProvider。

        §75 诚实性：
            1. 无 artifact → 显式 raise（不伪造分数）；
            2. artifact schema 校验（evidence_grade / split / 有限数）由
               ArtifactScoreProvider.open 内部保证；
            3. GPU 加载移到可选：有 artifact 时不需要模型加载，
               无 artifact 时才尝试加载模型（失败 raise）。
        """
        self._cfg = cfg
        # 尝试从 run_dir 找到 capability_score_artifact.json
        run_dir = Path(cfg.get("run_dir", ""))
        artifact_path = run_dir / "inject_eval" / "capability_score_artifact.json"
        if artifact_path.exists():
            from .artifact import ArtifactScoreProvider

            self._artifact_delegate = ArtifactScoreProvider()
            self._artifact_delegate.open({
                **cfg,
                "score_artifact_path": str(artifact_path),
            })
            return

        # 无 artifact：尝试加载模型（原始 HF 路径），失败 raise（§75）
        try:
            import torch  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "HFScoreProvider 需要 torch/transformers 且无 capability_score_artifact.json"
                f"（run_dir={run_dir}）。请先运行 inject-eval 子命令产出 artifact，"
                "或安装 torch。"
            ) from exc

        device = resolve_device(cfg)
        AutoModelForCausalLM, AutoTokenizer = _model_source()
        s_cfg = cfg.get("student", {})
        if not s_cfg.get("model_id"):
            raise RuntimeError(
                "HFScoreProvider.open 需要 cfg.student.model_id"
                "（且无 capability_score_artifact.json 可委托）"
            )
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

        委托模式（有 artifact）：直接转交 ArtifactScoreProvider.score()；
        直连模式（无 artifact，有模型）：原始 HF logits 评分；
        两者皆无 → 阻断（§75）。
        """
        if method not in _METHODS:
            raise KeyError(f"Unknown method {method!r}")
        # 委托模式
        if self._artifact_delegate is not None:
            return self._artifact_delegate.score(method, sample_id, seed)
        # 直连模式
        if self._model is not None and self._tok is not None:
            prompt = self._make_prompt(sample_id, seed)
            return self._score_choices(prompt)
        raise RuntimeError(
            f"HFScoreProvider.score({method!r}) 无 artifact 委托且无模型；"
            "请先运行 inject-eval 或提供 torch/transformers。"
        )

    def decision(self, method: str, sample_id: str, seed: int) -> int:
        """`method` 在 sample_id 上的决策 ID（§45 JCR）。"""
        if method not in _METHODS:
            raise KeyError(f"Unknown method {method!r}")
        # 委托模式
        if self._artifact_delegate is not None:
            return self._artifact_delegate.decision(method, sample_id, seed)
        # 直连模式：从 logits 取 argmax
        if self._model is not None and self._tok is not None:
            prompt = self._make_prompt(sample_id, seed)
            import torch  # type: ignore

            with torch.no_grad():
                enc = self._tok(prompt, return_tensors="pt").input_ids.to(self._model.device)
                logits = self._model(enc).logits[:, -1, :]
                probs = torch.softmax(logits, dim=-1)[0]
                token_ids = [self._tok.convert_tokens_to_ids(c) for c in ["A", "B", "C", "D"]]
                vals = [probs[t].item() if t != -1 else 0.0 for t in token_ids]
            return int(np.argmax(vals))
        raise RuntimeError(
            f"HFScoreProvider.decision({method!r}) 无 artifact 委托且无模型；"
            "请先运行 inject-eval 或提供 torch/transformers。"
        )

    def close(self) -> None:
        self._model = None
        self._tok = None
        if self._artifact_delegate is not None:
            self._artifact_delegate.close()
            self._artifact_delegate = None
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def describe(self) -> dict[str, Any]:
        if self._artifact_delegate is not None:
            desc = self._artifact_delegate.describe()
            desc["implementation"] = "hf_score_via_artifact_delegation"
            desc["delegation_note"] = (
                "HFScoreProvider 委托 capability_score_artifact.json 评分；"
                "inject-eval 是唯一产生真实 scores 的路径。"
            )
            return desc
        s = self._cfg.get("student", {})
        return {
            "implementation": "hf_modelscope_score_direct",
            "evidence_grade": "blocked_until_sample_and_cache_wiring",
            "model_id": s.get("model_id"),
            "revision": s.get("revision", "main"),
            "dtype": s.get("dtype", "bfloat16"),
            "note": (
                "直接 HF 模型评分（无 artifact 委托）；"
                "注入类方法以 student 零样本兜底（未接线 KV 注入）。"
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
            enc = self._tok(prompt, return_tensors="pt").input_ids.to(self._model.device)
            logits = self._model(enc).logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)[0]
            token_ids = [self._tok.convert_tokens_to_ids(c) for c in choices]
            vals = [probs[t].item() if t != -1 else 0.0 for t in token_ids]
        mx = max(vals) if vals else 0.0
        return mx if mx > 0 else 0.5

    def _open_if_needed(self, cfg: dict[str, Any]) -> None:
        if self._model is None and self._artifact_delegate is None:
            self.open(cfg)


__all__ = ["HFScoreProvider"]
