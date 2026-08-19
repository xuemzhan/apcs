"""Provider 抽象层（架构审查 P1-3 修复）。

═══════════════════════════════════════════════════════════════════════════════
**目的**：把「数据从哪里来」与「任务本身怎么算」彻底解耦。

修复前，runner 函数（如 `run_ridge_baseline`）内部直接调用
`_synth_calibration_set(...)` 生成数据；接真实 HF 模型时必须**改函数内脏**，
而非替换一个 provider。

修复后：
    - 每个 runner 接收一个 `KVProvider`（KV 校准/评估数据）和/或
      `ScoreProvider`（每方法得分）+ 可选 `TimingProvider`（系统成本）；
    - 仿真 / 真实 之间的切换 = 一个 cfg 字段：
        - `provider.kind: "synthetic"`  → `SyntheticProvider`（默认，numpy）
        - `provider.kind: "hf"`         → `HFProvider`（TODO(real-gpu)，需要 torch）
    - 同样 task 的 runner 不改一行业务逻辑即可切换。

**协议设计原则**：
    1. Provider 是**可 JSON 序列化的配置**（kv_cache_layer / checkpoint 等），
       而不是大活对象，跑前可保存一份到 run_dir；
    2. Provider 暴露**最小集合**的数值方法（get_calibration_kv / score），
       避免向 runner 泄漏 huggingface / tokenize 等细节；
    3. 任何缺失的 provider 字段都应在 `provider.open()` 时显式 raise，
       杜绝「静默回退到合成」。
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import contextlib
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Protocol

import numpy as np


# ---------------------------------------------------------------------------
# 协议定义（接口契约）
# ---------------------------------------------------------------------------
#
# Provider 协议不依赖具体实现；合成/真实两侧都只是实现这一组方法。
# 任何 runner 都应**只**依赖这些协议，**不能**直接依赖 SyntheticProvider 或
# HFProvider 的具体类。便于未来在不改 runner 的前提下替换实现。

@dataclass
class CalibrationSample:
    """单个校准样本：同一对 Teacher/Student 模型、同一 prompt 下的 KV 状态。

    Shape：
        kv_t: (n_t, S, H, D) float32 —— Teacher 一次 forward 抓取的 KV
        kv_s: (n_s, S, H, D) float32 —— 同 prompt 上 Student 自产 KV
               （注意：Student 自产 KV 与真实推理输入统一保留，因 §52.1
               合规要求不在主实验中让 Student 重新读 X；本样本用于
               mapper 校准与（transductive）attribution，属于 §22 拼装语料）
        sample_id: str —— 跨 run 可追溯
    """
    kv_t: np.ndarray
    kv_s: np.ndarray
    sample_id: str
    prompt: str | None = None
    split: str | None = None


class KVProvider(Protocol):
    """产出 (kv_t, kv_s) 校准 / 评估样本的数据源。

    接口最小化为两个生成器 — 校准与 held-out 评估分别从**母体不交的子集**
    取样（防数据泄漏，§32 bug-3 修复的语义）。
    """
    kind: str  # "synthetic" / "hf" / "random"

    def open(self, cfg: dict[str, Any]) -> None:
        """加载资源（如 model / tokenizer）。失败 → 显式 raise。

        不允许「加载失败 → 静默回退到合成」—— §75 诚实性。
        """
        ...

    def iter_calibration(
        self, n_samples: int, seed: int
    ) -> Iterator[CalibrationSample]:
        """生成 n_samples 个**校准**样本（用于 mapper.fit）。"""
        ...

    def iter_eval(
        self, n_samples: int, seed: int
    ) -> Iterator[CalibrationSample]:
        """生成 n_samples 个**held-out 评估**样本（用于 mapper.transform 后打分）。

        必须与 calibration 的样本**联合不交**（disjoint seed 子空间）。
        """
        ...

    def close(self) -> None:
        """释放资源（unload model / free GPU）。"""
        ...

    def describe(self) -> dict[str, Any]:
        """返回可序列化的描述（落到 run_dir/provider.json）。

        用于 §64/§65 metadata 与复现审计：审稿人应能根据 describe 字段
        知道本次实验**具体用了什么** Teacher/Student/数据集。
        """
        ...


class ScoreProvider(Protocol):
    """逐方法逐 sample 的得分数据源（§37 7 方法）。

    与 KVProvider 分离，因为很多实验（capability / system）只需要得分
    而不需要 KV；§37 七个方法 + 真实评估的对应关系是一组"调用 LLM/打分"
    的逻辑，独立于 KV 流。
    """
    kind: str

    def score(self, method: str, sample_id: str, seed: int) -> float:
        """返回 method 在 sample_id 上的得分（0–1 准确率或 0–100 指标）。

        method ∈ {"student", "teacher", "text", "ridge", "base_only",
                  "base_plus_adv", "full_apcs"}（§37 报口径）。
        """
        ...

    def decision(self, method: str, sample_id: str, seed: int) -> int:
        """返回方法在 sample_id 上的决策 ID（ranker / judge / multi-choice）。

        §45 JCR 计算用。"""
        ...

    def describe(self) -> dict[str, Any]: ...


class TimingProvider(Protocol):
    """系统耗时数据源（§38 / §49 / §53）。

    真实实现：torch.cuda.synchronize + time.perf_counter，
    warmup + ≥10 repeats + 取 P50/P95（§49）。
    """
    kind: str

    def measure(self, ctx: int, seed: int) -> dict[str, float]:
        """返回各组件耗时（ms），键：teacher_prefill / student_prefill / map /
        load / query。"""
        ...

    def measure_vram(self) -> int:
        """返回当前 Teacher+ Student 权重加 KV 缓存的总显存占用（MB）。"""
        ...

    def describe(self) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
# 工厂：按 cfg["provider"] 选 provider（单点切换）
# ---------------------------------------------------------------------------


def _provider_missing_signal_raise(provider_kind: str, req: str) -> None:
    """Provider 缺失/未接线时的统一报错入口。

    设计原则：失败 → 显式 raise，**绝不**静默回退到合成。
    §75 诚实性 + 杜绝「偷偷切回合成导致结果看似可复现」的退化路径。
    """
    raise RuntimeError(
        f"Provider '{provider_kind}' 不可用：{req}。"
        "若你 **故意** 要回退到合成模式，请在 cfg 中显式设 "
        "`provider.kind: synthetic`，让该选择被记录在 run_dir/provider.json。"
    )


def _build_kv_provider(cfg: dict[str, Any]) -> KVProvider:
    """构造 KVProvider（按 cfg["provider"]["kind"] 分流）。"""
    kind = cfg.get("provider", {}).get("kv", "synthetic").lower()
    if kind == "synthetic":
        from .synthetic_kv import SyntheticKVProvider

        return SyntheticKVProvider(seed=int(cfg.get("seeds", [0])[0]))
    if kind == "hf":
        # 真实 HF KVProvider（modelscope 源；GPU 不可用时 open() 显式 raise）
        from .hf_kv import HFKVProvider

        return HFKVProvider()
    raise ValueError(f"Unknown KV provider kind: {kind!r}")


def _build_score_provider(cfg: dict[str, Any]) -> ScoreProvider:
    """构造 ScoreProvider。"""
    kind = cfg.get("provider", {}).get("score", "synthetic").lower()
    if kind == "synthetic":
        from .synthetic_score import SyntheticScoreProvider

        return SyntheticScoreProvider()
    if kind == "hf":
        from .hf_score import HFScoreProvider

        return HFScoreProvider()
    raise ValueError(f"Unknown Score provider kind: {kind!r}")


def _build_timing_provider(cfg: dict[str, Any]) -> TimingProvider:
    """构造 TimingProvider。"""
    kind = cfg.get("provider", {}).get("timing", "synthetic").lower()
    if kind == "synthetic":
        from .synthetic_timing import SyntheticTimingProvider

        return SyntheticTimingProvider()
    if kind == "hf":
        from .hf_timing import HFTimingProvider

        return HFTimingProvider()
    raise ValueError(f"Unknown Timing provider kind: {kind!r}")


# ---------------------------------------------------------------------------
# 通用工具：把 provider 描述 & 当前使用状态写入 run_dir
# ---------------------------------------------------------------------------


def write_provider_manifest(
    run_dir: Path,
    *,
    kv: KVProvider | None = None,
    score: ScoreProvider | None = None,
    timing: TimingProvider | None = None,
) -> dict[str, Any]:
    """把三家 provider 的 describe 合并写到 `<run_dir>/provider.json`。

    用途：复现审计 —— 任何人打开 run 目录都能立刻知道这次实验「究竟用了
    什么 provider」，避免「以为是真实实验实际是合成」之类的事故。
    与 §64 metadata 互补，但粒度更专业（专门记录数据/评分来源）。
    """
    manifest: dict[str, Any] = {}
    for name, p in (("kv", kv), ("score", score), ("timing", timing)):
        if p is None:
            continue
        entry = {"kind": p.kind}
        try:
            entry.update(p.describe())
        except Exception as e:  # noqa: BLE001
            # 描述失败不应阻断 task 本身；只记 warn
            entry["describe_error"] = f"{type(e).__name__}: {e}"
        manifest[name] = entry
    (run_dir / "provider.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def write_provider_selection(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """不打开模型，仅记录本次配置选择；适合 CLI 在任务开始时调用。"""
    selected = cfg.get("provider", {}) or {}
    manifest = {
        name: {
            "kind": str(selected.get(name, "synthetic")).lower(),
            "status": "configured_not_opened",
        }
        for name in ("kv", "score", "timing")
    }
    (run_dir / "provider.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


# ---------------------------------------------------------------------------
# 便捷门面：runner 调用者一般只需要这两行
# ---------------------------------------------------------------------------


def open_providers(
    cfg: dict[str, Any],
    *,
    need: tuple[str, ...] = ("kv", "score", "timing"),
) -> dict[str, Any]:
    """按 cfg 一次性构造并 open 全部需要的 provider。

    返回 dict {"kv": ..., "score": ..., "timing": ...}，调用方按需取用。
    close 责任由 caller 负责（建议配合 `with providers_block(cfg, need=...)`）。
    """
    builders = {
        "kv": _build_kv_provider,
        "score": _build_score_provider,
        "timing": _build_timing_provider,
    }
    out: dict[str, Any] = {}
    try:
        for name in need:
            if name not in builders:
                raise ValueError(f"Unknown provider channel: {name}")
            p = builders[name](cfg)
            if hasattr(p, "open"):
                p.open(cfg)
            out[name] = p
    except Exception:
        # 多 provider 打开到一半失败时，必须释放已经加载的模型/GPU 缓存。
        for opened in reversed(list(out.values())):
            if hasattr(opened, "close"):
                opened.close()
        raise
    return out


@contextmanager
def providers_ctx(
    cfg: dict[str, Any],
    *,
    need: tuple[str, ...] = ("kv", "score", "timing"),
):
    """与 open_providers 配对的 context manager：出 with 自动 close。

    用法：
        with providers_ctx(cfg, need=("kv",)) as providers:
            kvp = providers["kv"]
            for s in kvp.iter_calibration(...)
                ...
    """
    from contextlib import ExitStack

    providers = open_providers(cfg, need=need)
    with ExitStack() as stack:
        for p in providers.values():
            if hasattr(p, 'close'):
                stack.callback(p.close)
        yield providers
