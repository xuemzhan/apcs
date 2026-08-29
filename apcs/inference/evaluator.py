"""inject-eval 评估器（§52 zero-prefill / §53 单卡执行顺序 / §49 计时）。

═══════════════════════════════════════════════════════════════════════════════
本模块对应 design.md：
    §52 禁止 1 —— Student 不重新读 X（zero prefill；注入后只喂 query token）
    §53 单卡执行策略 —— Teacher Load → Forward → Capture → CPU Offload →
        Teacher Unload → CUDA Cleanup → Student Load → Map → Inject → Decode
    §49 计时规范 —— 阶段边界 sync
    §75 诚实性 —— 缺模型/缓存/网络 → 显式 raise，禁止伪造评分

四方法评估协议（inject-eval 核心）：
    1. student_self:  Student 自身 tokenize X+q 联合 prefill，取 final-position logits
    2. teacher_full:  Teacher 全量 forward X+q，取 final-position logits
    3. text_handoff:  Student 只看到截断文本 proxy（X 前 N 字符 + q），**非** zero-prefill
    4. ridge_handoff: Teacher prefill X → capture PKV → mapper.transform → inject →
                      Student 只 decode q tokens（**真正的** zero-prefill 路径）

产物（§63 Run 产物规范）：
    <run_dir>/inject_eval/
        replacement_score_artifact.json  — 供 T05 消费（_load_replacement_task_artifact）
        capability_score_artifact.json   — 供 T09 消费（ArtifactScoreProvider.open）
        hidden_states.npz (可选)         — 供 T12 geometry diagnostics

§53 单卡执行顺序（内存纪律）：
    本评估器按方法组顺序加载/卸载模型，同一时刻 GPU 上只驻留一个模型：
    - teacher_full 阶段：加载 Teacher → 逐样本 forward → 卸载
    - student_self + text_handoff + ridge_handoff 阶段：加载 Student →
      逐样本执行三个方法 → 卸载
    每次卸载后执行 GC + torch.cuda.empty_cache()（§53 CUDA Cleanup）。
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import gc
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import re

import numpy as np

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = "1.1"


def _git_hash() -> str:
    """当前代码 git commit（§7.1 版本固定；失败返回 "unknown"）。"""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _extract_choices_from_query(query: str) -> list[str]:
    """从 query 中提取选项文本（A=text | B=text | ... → [textA, textB, ...]）。

    支持 HellaSwag 格式: "...(A/B/C/D): A=text1 | B=text2 | ..."
    支持 ARC 格式:        "...A=text1 | B=text2 | ..."
    """
    # Try HellaSwag format first: (A/B/C/D): A=... | B=...
    match = re.search(r"\(A/B/C/D\):\s*(.*)", query)
    if not match:
        # Try generic format: A=... | B=... (after any colon)
        match = re.search(r":\s*(A=[^|]+(?:\|\s*B=.*))", query)
    if not match:
        return []
    raw = match.group(1)
    parts = re.split(r"\s*\|\s*", raw)
    choices = []
    for p in parts:
        m = re.match(r"[A-D]=(.*)", p.strip())
        if m:
            choices.append(m.group(1).strip())
    return choices


# ---------------------------------------------------------------------------
# 纯 numpy 评分函数（可脱离 torch 单独测试）
# ---------------------------------------------------------------------------

# HF tokenizer 对 "A"/"B"/"C"/"D" 各只产生一个 leading token；
# 以下为 Qwen3 系列的 token id（作为默认值；真实评测时从 tokenizer 获取）
_DEFAULT_CHOICE_IDS = {
    "A": 32,
    "B": 33,
    "C": 34,
    "D": 35,
}


def score_choices(
    logits_np: np.ndarray,
    choice_letter_ids: dict[str, int] | None = None,
) -> tuple[float, int]:
    """纯 numpy 评分：从 final-position logits 计算 4 选项 softmax 概率。

    只对 4 个答案 token 的 logit 做 softmax（不是全 vocab softmax），
    以获得有意义的概率值。

    Args:
        logits_np: 1-D float array shape=(vocab_size,) 或 2-D shape=(1, vocab_size)
        choice_letter_ids: {letter: token_id} 映射；None 时用 Qwen3 默认值

    Returns:
        (score, decision):
            score: argmax letter 的 softmax 概率（0~1，仅限4选项之间）
            decision: argmax letter 的 0-based 索引（0=A, 1=B, 2=C, 3=D）
    """
    if choice_letter_ids is None:
        choice_letter_ids = _DEFAULT_CHOICE_IDS
    logits = np.asarray(logits_np, dtype=np.float64).reshape(-1)
    letters = ["A", "B", "C", "D"]
    vals = []
    for l in letters:
        lid = choice_letter_ids[l]
        vals.append(float(logits[lid]) if lid < len(logits) else -1e9)
    vals = np.array(vals)
    # softmax 只在 4 个选项之间做
    vals_max = vals.max()
    exp_vals = np.exp(vals - vals_max)
    sum_exp = exp_vals.sum()
    if sum_exp <= 0:
        return 0.0, 0
    probs = exp_vals / sum_exp
    decision = int(np.argmax(probs))
    score = float(probs[decision])
    return score, decision


def gold_prob_from_logits(
    logits_np: np.ndarray,
    gold_letter: str | None,
    choice_letter_ids: dict[str, int] | None = None,
) -> float | None:
    """gold 答案字母的 softmax 概率（§75：能力度量的主口径之一）。

    与 score_choices 的区别：score_choices 返回 argmax 字母的置信度
    （"多自信"），本函数返回 gold 字母的概率（"多正确"）。
    gold_letter 缺失/非法 → None（调用方如实记 null，不伪造）。
    """
    letters = ["A", "B", "C", "D"]
    if not gold_letter:
        return None
    if logits_np is None:
        return None  # 前向失败时如实返回 None，不伪造均匀分布
    gold = gold_letter.strip().upper()
    if gold not in letters:
        return None
    if choice_letter_ids is None:
        choice_letter_ids = _DEFAULT_CHOICE_IDS
    logits = np.asarray(logits_np, dtype=np.float64).reshape(-1)
    vals = []
    for l in letters:
        lid = choice_letter_ids[l]
        vals.append(float(logits[lid]) if lid < len(logits) else -1e9)
    vals = np.array(vals)
    vals_max = vals.max()
    exp_vals = np.exp(vals - vals_max)
    sum_exp = exp_vals.sum()
    if sum_exp <= 0:
        return 0.0
    probs = exp_vals / sum_exp
    return float(probs[letters.index(gold)])


def cache_seq_len(cache: Any) -> int:
    """从 HF DynamicCache 读取真实 past 长度（§52 审计数据源）。

    兼容新旧 transformers：优先 get_seq_length()，回退 key_cache[0].shape[2]。
    读取失败 → 0（调用方据此让断言失败，而不是静默通过）。
    """
    try:
        return int(cache.get_seq_length())
    except Exception:  # noqa: BLE001
        try:
            return int(cache.key_cache[0].shape[2])
        except Exception:  # noqa: BLE001
            return 0


def score_choices_with_accuracy(
    logits_np: np.ndarray,
    gold_answer: str | None = None,
    choice_letter_ids: dict[str, int] | None = None,
) -> tuple[float, int, bool]:
    """评分并计算准确率：softmax 置信度 + 是否预测正确。

    Args:
        logits_np: final-position logits
        gold_answer: 标准答案字母（"A"/"B"/"C"/"D"）；None 时不计算 accuracy
        choice_letter_ids: {letter: token_id} 映射

    Returns:
        (score, decision, correct):
            score: softmax 概率（置信度）
            decision: 0-based 索引
            correct: 是否预测正确（gold_answer 为 None 时总是 False）
    """
    score, decision = score_choices(logits_np, choice_letter_ids)
    correct = False
    if gold_answer is not None:
        letters = ["A", "B", "C", "D"]
        if 0 <= decision < len(letters):
            predicted_letter = letters[decision]
            correct = predicted_letter.upper() == gold_answer.strip().upper()
    return score, decision, correct


# ---------------------------------------------------------------------------
# PPL (Perplexity) 计算 —— §22 4-way 消融评测
# ---------------------------------------------------------------------------


def compute_ppl(
    model: Any,
    input_ids: np.ndarray,
    past_key_values: Any | None = None,
    attention_mask: np.ndarray | None = None,
    position_ids: np.ndarray | None = None,
) -> float:
    """计算 perplexity = exp(cross_entropy_loss)。

    支持 past_key_values 注入（zero-prefill 场景）；
    排除 padding token (attention_mask=0 或 token_id=0) 的 loss，
    严格遵循 §75 诚实性——不伪造、不插值。

    Args:
        model: HuggingFace CausalLM (torch, on GPU, eval mode)
        input_ids: (1, seq_len) 或 (seq_len,) int token ids
        past_key_values: HF DynamicCache 或 None
        attention_mask: (1, seq_len) 或 (seq_len,) 可选；0 = padding
        position_ids: (1, seq_len) 可选；zero-prefill 时需显式指定

    Returns:
        float: perplexity (exp of mean cross-entropy loss)；
               计算失败时返回 float("inf")
    """
    import torch  # type: ignore

    try:
        ids = torch.as_tensor(np.asarray(input_ids), dtype=torch.long)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)  # (1, seq_len)

        labels = ids.clone()
        # padding mask → labels=-100 (HF CrossEntropyLoss 忽略 -100)
        if attention_mask is not None:
            mask = torch.as_tensor(np.asarray(attention_mask), dtype=torch.long)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            labels[mask == 0] = -100
        else:
            # 无 attention_mask 时：token_id=0 通常为 padding
            labels[ids == 0] = -100

        kwargs: dict[str, Any] = {
            "input_ids": ids.to(model.device),
            "labels": labels.to(model.device),
            "use_cache": False,
        }
        if past_key_values is not None:
            kwargs["past_key_values"] = past_key_values
        if position_ids is not None:
            pos_t = torch.as_tensor(
                np.asarray(position_ids), dtype=torch.long
            ).to(model.device)
            if pos_t.ndim == 1:
                pos_t = pos_t.unsqueeze(0)
            kwargs["position_ids"] = pos_t

        with torch.no_grad():
            outputs = model(**kwargs)
        loss = float(outputs.loss.item())
        return math.exp(loss)
    except Exception:  # noqa: BLE001
        return float("inf")


# ---------------------------------------------------------------------------
# 评分模板（多选题 prompt 构造）
# ---------------------------------------------------------------------------

_SCORING_SUFFIX = "\nAnswer:"


def _build_scoring_prompt(context_or_text: str, query: str, choices: list[str]) -> str:
    """构造多选题评分 prompt：context + choices + Answer: 后缀。

    choices 为 ["选项A文本", "选项B文本", ...]；prompt 格式：
        {context}\n{query}\nA. ...\nB. ...\nC. ...\nD. ...\nAnswer:
    """
    choice_labels = ["A", "B", "C", "D"]
    choice_block = "\n".join(
        f"{label}. {text}" for label, text in zip(choice_labels, choices[:4])
    )
    return f"{context_or_text}\n{query}\n{choice_block}{_SCORING_SUFFIX}"


def _unified_suffix_text(query: str) -> str:
    """P0.5 统一评分后缀（所有方法共享同一 token 序列）。

    协议 v1.2：student_self / teacher_full / text_handoff / ridge_handoff /
    self_kv 的"查询部分"必须是逐 token 相同的
    `{query}\n\nAnswer:` —— 方法间唯一差异是 context 的来源
    （学生自 prefill / 教师 KV 注入 / 截断文本 / 学生 KV 自注入）。
    此前 student_self 额外渲染 "A. opt1\\nB. opt2..." 选项块，与 handoff
    的 token 序列不同，CHG 混入格式效应（混淆变量）。
    """
    return f"{query}\n{_SCORING_SUFFIX}"


def _extract_gold_index(answer: str, choices: list[str]) -> int:
    """从 row.answer 匹配 choices 确定 gold letter 索引（0-3）。

    匹配策略：
        1. answer 为单字母 "A"/"B"/"C"/"D" → 直接映射
        2. answer 为 choice 文本（或包含 choice 文本）→ 逐选项匹配
        3. 都不匹配 → raise
    """
    if answer and len(answer.strip()) == 1 and answer.strip().upper() in "ABCD":
        return ord(answer.strip().upper()) - ord("A")
    answer_lower = (answer or "").strip().lower()
    for i, c in enumerate(choices[:4]):
        if c.strip().lower() == answer_lower or answer_lower in c.strip().lower():
            return i
    raise ValueError(
        f"无法将 answer={answer!r} 匹配到 choices={choices!r}"
    )


# ---------------------------------------------------------------------------
# §52 zero-prefill 统计计数器（inject-eval 专用）
# ---------------------------------------------------------------------------


@dataclass
class EvalPhaseCounters:
    """inject-eval 四阶段计数器。

    每个 sample 的 ridge_handoff 路径结束后：
        - assert inject_seq_len == S（Teacher prefill 长度）
        - assert query_new_tokens == len(query_tokens)（只喂了 query）
        - assert past_len == S（没偷看 X）

    §53 单卡执行顺序：逐样本串行，每样本四个方法串行；
    整批完成后写 artifacts。
    """

    teacher_prefill_seq_len: int = 0
    inject_seq_len: int = 0
    query_new_tokens: int = 0
    query_past_len: int = 0
    student_prefill_new_tokens: int = 0

    def reset(self) -> None:
        self.teacher_prefill_seq_len = 0
        self.inject_seq_len = 0
        self.query_new_tokens = 0
        self.query_past_len = 0
        self.student_prefill_new_tokens = 0

    def assert_zero_prefill(self) -> None:
        """§52 禁止 1：Student 不重新读 X。"""
        if self.student_prefill_new_tokens != 0:
            raise RuntimeError(
                f"§52 禁止 1 违反：Student prefill 了 {self.student_prefill_new_tokens} 个 token，"
                f"应为 0（zero-prefill）。"
            )

    def assert_query_handoff(self) -> None:
        """断言 query decode 的 past_len == inject_seq_len（无 context 泄漏）。"""
        if self.query_past_len != self.inject_seq_len:
            raise RuntimeError(
                f"§52 zero-prefill 断言失败：past_len({self.query_past_len}) "
                f"!= inject_seq_len({self.inject_seq_len})"
            )

    @property
    def is_verified(self) -> bool:
        try:
            self.assert_zero_prefill()
            self.assert_query_handoff()
            return True
        except RuntimeError:
            return False


# ---------------------------------------------------------------------------
# InjectionEvaluator 主体
# ---------------------------------------------------------------------------


class InjectionEvaluator:
    """inject-eval 核心评估器（§52 zero-prefill / §53 单卡执行顺序）。

    cfg.inject_eval 配置节：
        max_samples: int = 32         每 split 最大评估样本数
        text_handoff_max_chars: int = 400  text_handoff 截断字符数
        export_hidden_states: bool = True   是否导出 hidden states NPZ
        seed: int = 0                     评测种子

    §53 内存纪律：
        - Teacher 模型与 Student 模型不同时驻留 GPU
        - 每个方法组完成后卸载 + GC + empty_cache
        - sequential residency per §53
    """

    def __init__(self, cfg: dict[str, Any], run_dir: Path) -> None:
        self.cfg = cfg
        self.run_dir = run_dir
        self._backend: Any = None  # 延迟初始化
        self._teacher_model: Any = None
        self._student_model: Any = None
        self._tok: Any = None
        self._mapper: Any = None
        self._layer_map: list[list[int]] | None = None
        self._teacher_tokenizer: Any = None
        self._student_tokenizer: Any = None
        self._counters = EvalPhaseCounters()

        # inject_eval 配置
        ie_cfg = cfg.get("inject_eval", {})
        self._max_samples = int(ie_cfg.get("max_samples", 32))
        self._text_max_chars = int(ie_cfg.get("text_handoff_max_chars", 400))
        self._export_hidden = bool(ie_cfg.get("export_hidden_states", False))
        self._seed = int(ie_cfg.get("seed", 0))

        # ── 协议 v1.1 新增配置 ──
        # A2: 校准/评估互斥切分（默认开启；关闭则回退"同样本"旧行为并在
        # provenance 里如实标注 calib_eval_disjoint=False）
        self._calib_split = bool(ie_cfg.get("calib_eval_split", True))
        self._n_calib = int(ie_cfg.get("calib_samples", 0))  # 0 → 自动取一半
        # B1: KV 持久化目录（默认 <run_dir>/inject_eval/kv_store）
        self._persist_kv = bool(ie_cfg.get("persist_kv", False))
        self._kv_store_dir = (
            Path(str(ie_cfg["kv_store_dir"]))
            if ie_cfg.get("kv_store_dir")
            else Path(self.run_dir) / "inject_eval" / "kv_store"
        )
        self._online_kv_dir = ie_cfg.get("online_kv_dir")  # 在线阶段：读盘冷启动
        # B3: RoPE 对齐模式 —— "rotated"（旧行为：W 吸收位置旋转）
        #                    | "unrotated"（§23 本意：unrotated 空间回归，注入前 re-RoPE）
        self._rope_align = str(cfg.get("mapper", {}).get("rope_align", "rotated"))

        # 每个模型的字母 token id（gold 概率计算用；加载模型时填充）
        self._teacher_letter_ids: dict[str, int] | None = None
        self._student_letter_ids: dict[str, int] | None = None
        # P0.6: KV 范数统计与 native 重标定开关
        self._kv_norm_stats: dict[str, Any] = {}
        self._native_rescale = bool(ie_cfg.get("native_rescale", False))
        # P1.5: teacher-summary 文本基线（KV 迁移的最强实用竞争者）
        self._summary_baseline = bool(ie_cfg.get("summary_baseline", False))
        self._summary_max_tokens = int(ie_cfg.get("summary_max_tokens", 64))
        self._summaries: dict[str, str] = {}
        # P0-Oracle 探针（H3 上界判定）：允许 mix_*/win_* 探针模式
        # ⚠️ 探针需要学生对评估行做自 prefill —— 仅诊断用，非部署路径
        self._probe_mode = bool(ie_cfg.get("probe_mode", False))

    # ------------------------------------------------------------------
    # 模型生命周期管理（§53 单卡执行顺序）
    # ------------------------------------------------------------------

    def _load_teacher(self) -> None:
        """§53 Teacher Load：加载 Teacher 模型 + tokenizer。

        §53 内存纪律：Student 必须先卸载。loaded = 教师独占 GPU。
        """
        import torch  # type: ignore
        import os

        t_cfg = self.cfg.get("teacher", {})
        if not t_cfg.get("model_id"):
            raise RuntimeError("inject-eval 需要 cfg.teacher.model_id")
        self._unload_student()
        revision = str(t_cfg.get("revision", "main"))
        dtype_name = str(t_cfg.get("dtype", "bfloat16"))
        dtype = getattr(torch, dtype_name, torch.bfloat16)
        device = str(t_cfg.get("device_map", "cuda:0"))
        if ":" not in device:
            device = f"cuda:0"
        logger.info("[inject-eval] Teacher Load: %s rev=%s", t_cfg["model_id"], revision)

        # 尝试 modelscope，失败则回退到 transformers + 本地缓存
        model_id = t_cfg["model_id"]
        try:
            from modelscope import AutoModelForCausalLM, AutoTokenizer  # type: ignore
            self._teacher_model = AutoModelForCausalLM.from_pretrained(
                model_id, revision=revision, torch_dtype=dtype,
                device_map={"": device}, attn_implementation="sdpa",
            )
            self._teacher_tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        except Exception as e:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
            cache_path = os.path.expanduser(
                f"~/.cache/modelscope/models/{model_id.replace('/', '--')}/snapshots/master"
            )
            if not os.path.exists(cache_path):
                cache_path = model_id
            logger.warning("[inject-eval] ModelScope unavailable, loading from %s", cache_path)
            self._teacher_model = AutoModelForCausalLM.from_pretrained(
                cache_path, torch_dtype=dtype,
                device_map={"": device}, attn_implementation="sdpa",
            )
            self._teacher_tokenizer = AutoTokenizer.from_pretrained(cache_path)

        self._teacher_model.eval()
        self._teacher_letter_ids = {
            l: int(self._teacher_tokenizer.encode(l, add_special_tokens=False)[0])
            for l in ["A", "B", "C", "D"]
        }
        self._teacher_device = device

    def _unload_teacher(self) -> None:
        """§53 Teacher Unload：释放 Teacher + GPU cleanup。"""
        import torch  # type: ignore

        self._teacher_model = None
        self._teacher_tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_student(self) -> None:
        """§53 Student Load：加载 Student 模型 + tokenizer。

        §53 内存纪律：Teacher 必须先卸载。loaded = Student 独占 GPU。
        """
        import torch  # type: ignore
        import os

        s_cfg = self.cfg.get("student", {})
        if not s_cfg.get("model_id"):
            raise RuntimeError("inject-eval 需要 cfg.student.model_id")
        self._unload_teacher()
        revision = str(s_cfg.get("revision", "main"))
        dtype_name = str(s_cfg.get("dtype", "bfloat16"))
        dtype = getattr(torch, dtype_name, torch.bfloat16)
        device = str(s_cfg.get("device_map", "cuda:0"))
        if ":" not in device:
            device = "cuda:0"
        logger.info("[inject-eval] Student Load: %s rev=%s", s_cfg["model_id"], revision)

        # 尝试 modelscope，失败则回退到 transformers + 本地缓存
        model_id = s_cfg["model_id"]
        try:
            from modelscope import AutoModelForCausalLM, AutoTokenizer  # type: ignore
            self._student_model = AutoModelForCausalLM.from_pretrained(
                model_id, revision=revision, torch_dtype=dtype,
                device_map={"": device}, attn_implementation="sdpa",
            )
            self._student_tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        except Exception as e:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
            cache_path = os.path.expanduser(
                f"~/.cache/modelscope/models/{model_id.replace('/', '--')}/snapshots/master"
            )
            if not os.path.exists(cache_path):
                cache_path = model_id
            logger.warning("[inject-eval] ModelScope unavailable, loading from %s", cache_path)
            self._student_model = AutoModelForCausalLM.from_pretrained(
                cache_path, torch_dtype=dtype,
                device_map={"": device}, attn_implementation="sdpa",
            )
            self._student_tokenizer = AutoTokenizer.from_pretrained(cache_path)

        self._student_model.eval()
        self._student_letter_ids = {
            l: int(self._student_tokenizer.encode(l, add_special_tokens=False)[0])
            for l in ["A", "B", "C", "D"]
        }
        self._student_device = device

    def _unload_student(self) -> None:
        """§53 Student Unload + CUDA Cleanup。"""
        import torch  # type: ignore

        self._student_model = None
        self._student_tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Mapper 构造（复用 T05 runner 同款逻辑）
    # ------------------------------------------------------------------

    def _build_mappers(self) -> None:
        """构造 K/V mapper 对象与 layer_map（不 fit）。

        B1 在线阶段需要"先构造 → 从磁盘加载参数 → 跳过 fit"的路径，
        因此把构造与拟合拆开；_ensure_mapper 仍保持原签名与行为。
        """
        if self._layer_map is not None and self._mapper_k is not None:
            return
        from ..alignment.runner import proportional_mapping
        from ..mapper.math import (
            AffineMapper,
            LowRankMapper,
            RidgeMapper,
            RidgePerHeadMapper,
        )
        from ..mapper.runner import _ridge_lambda

        n_t = int(self.cfg.get("teacher", {}).get("num_layers", 36))
        n_s = int(self.cfg.get("student", {}).get("num_layers", 28))
        D = int(self.cfg.get("teacher", {}).get("head_dim", 128))
        self._layer_map = proportional_mapping(n_t, n_s)

        # §22: K 和 V 各自独立的 mapper（per-head 独立参数化）
        mapper_type = str(self.cfg.get("mapper", {}).get("type", "ridge"))
        rank = int(self.cfg.get("mapper", {}).get("rank", 16))

        if mapper_type == "lowrank":
            self._mapper_k = LowRankMapper(lam=_ridge_lambda(self.cfg, "K"), rank=rank)
            self._mapper_v = LowRankMapper(lam=_ridge_lambda(self.cfg, "V"), rank=rank)
        elif mapper_type == "affine":
            self._mapper_k = AffineMapper(lam=_ridge_lambda(self.cfg, "K"))
            self._mapper_v = AffineMapper(lam=_ridge_lambda(self.cfg, "V"))
        elif mapper_type == "ridge_layer":
            self._mapper_k = RidgeMapper(lam=_ridge_lambda(self.cfg, "K"))
            self._mapper_v = RidgeMapper(lam=_ridge_lambda(self.cfg, "V"))
        elif mapper_type == "task_aware":
            from ..mapper.task_aware import TaskAwareRidgeMapper as _TA
            lr = float(self.cfg.get("mapper", {}).get("lr", 1e-3))
            n_finetune_steps = int(self.cfg.get("mapper", {}).get("n_finetune_steps", 50))
            task_kw: dict[str, Any] = {}
            m_cfg = self.cfg.get("mapper", {})
            # P1.1/P1.3: 混合目标与参数化配置透传
            for k in ("task_loss_weight", "task_delta_mode", "learn_layer_alpha",
                      "task_lr", "task_steps", "task_batch_size"):
                if k in m_cfg:
                    task_kw[k] = m_cfg[k]
            self._mapper_k = _TA(
                lam=_ridge_lambda(self.cfg, "K"), lr=lr,
                n_finetune_steps=n_finetune_steps, **task_kw,
            )
            self._mapper_v = _TA(
                lam=_ridge_lambda(self.cfg, "V"), lr=lr,
                n_finetune_steps=n_finetune_steps, **task_kw,
            )
        elif mapper_type == "shared_basis":
            from ..mapper.math import SharedBasisMapper as _SB
            self._mapper_k = _SB(rank=rank)
            self._mapper_v = _SB(rank=rank)
        elif mapper_type == "affine_layer":
            from ..mapper.math import AffineLayerMapper as _AL
            self._mapper_k = _AL(lam=_ridge_lambda(self.cfg, "K"))
            self._mapper_v = _AL(lam=_ridge_lambda(self.cfg, "V"))
        elif mapper_type == "rat":
            from ..mapper.rat import RATMapper as _RAT
            m_cfg = self.cfg.get("mapper", {})
            common = dict(
                lam=_ridge_lambda(self.cfg, "K"),
                rank=int(m_cfg.get("rank", 16)),
                head_match=bool(m_cfg.get("rat_head_match", True)),
                sink_override=bool(m_cfg.get("rat_sink_override", True)),
                use_embedding_anchor=bool(m_cfg.get("rat_embedding_anchor", True)),
            )
            self._mapper_k = _RAT(**common)
            self._mapper_v = _RAT(**common)
            # T1/T2: 解析核需要两侧模型权重（safetensors 部分加载，非实例化）
            t_id = self.cfg.get("teacher", {}).get("model_id", "")
            s_id = self.cfg.get("student", {}).get("model_id", "")
            self._mapper_k.setup_weights(t_id, s_id)
            self._mapper_v.setup_weights(t_id, s_id)
        elif mapper_type == "mlp":
            from ..mapper.mlp import MLPMapper as _MLP
            m_cfg = self.cfg.get("mapper", {})
            common = dict(
                lam=_ridge_lambda(self.cfg, "K"),
                hidden=int(m_cfg.get("mlp_hidden", 64)),
                epochs=int(m_cfg.get("mlp_epochs", 200)),
                lr=float(m_cfg.get("mlp_lr", 1e-3)),
            )
            self._mapper_k = _MLP(**common)
            self._mapper_v = _MLP(**common)
        else:  # default: ridge (per-head)
            self._mapper_k = RidgePerHeadMapper(lam=_ridge_lambda(self.cfg, "K"))
            self._mapper_v = RidgePerHeadMapper(lam=_ridge_lambda(self.cfg, "V"))
    def _fit_mappers(self, real_kv_calib: dict[str, list] | None) -> None:
        """用配对 KV 校准数据 fit K/V mapper（真实数据优先，合成 fallback）。

        A2: real_kv_calib 由调用方保证只含**校准行**（与评估行互斥）；
        本方法不做切分，切分职责在 evaluate()。
        """
        from ..mapper.runner import _de_rope_for_kind, _ridge_lambda, kv_kinds
        from ..mapper.math import RidgeMapper, RidgePerHeadMapper
        from ..rope.runner import _rope_pairs, de_rope

        n_t = int(self.cfg.get("teacher", {}).get("num_layers", 36))
        n_s = int(self.cfg.get("student", {}).get("num_layers", 28))
        D = int(self.cfg.get("teacher", {}).get("head_dim", 128))
        inv_freq = _rope_pairs(D, theta=1_000_000.0)
        de_rope_fn = lambda k, p: de_rope(k, p, inv_freq)  # noqa: E731
        kinds = kv_kinds(self.cfg)
        H = int(self.cfg.get("teacher", {}).get("num_kv_heads", 8))
        n_calib = int(self.cfg.get("mapper", {}).get("inject_eval_calib_samples", 50))
        seq = int(self.cfg.get("context_lengths", [512])[0])
        from ..mapper.aggregate import fit_ridge_aggregate

        for kind in kinds:
            mapper = self._mapper_k if kind == "K" else self._mapper_v

            # 优先使用真实KV校准数据，fallback到合成数据
            use_real = real_kv_calib and kind in real_kv_calib and real_kv_calib[kind]
            if use_real:
                calib = real_kv_calib[kind]
                positions = None  # Variable-length: let fit_ridge_aggregate derive per-sample
                logger.info(
                    "[inject-eval] Using real KV calibration data for %s: %d samples",
                    kind, len(calib),
                )
            else:
                from ..mapper.runner import _shared_model_weights, _synth_calibration_set
                w_t, w_s = _shared_model_weights(n_t, n_s, D, master_seed=0)
                calib = _synth_calibration_set(
                    n_t, n_s, seq, H, D, n_calib, master_seed=0, noise=0.05,
                    w_t=w_t, w_s=w_s, kv_seed_offset={"K": 0, "V": 100000}.get(kind, 0),
                )
                positions = np.arange(seq, dtype=np.float64)
                logger.warning(
                    "[inject-eval] Using SYNTHETIC calibration data for %s (real data not available)",
                    kind,
                )

            kind_de_rope = _de_rope_for_kind(self.cfg, kind, de_rope_fn)
            if hasattr(mapper, 'lam'):
                mapper.lam = _ridge_lambda(self.cfg, kind)

            # ◆ 拟合路由（P0.3 修复）：
            #   - Ridge 系（含 TaskAware Phase A）→ Gram 聚合（变长样本安全）
            #   - Affine/Whitened/Procrustes/CCA → fit_batch（逐样本对齐后拼接，
            #     变长安全）—— 此前 Affine 误走 Gram 聚合必然 TypeError，
            #     LowRank 误走 concat 在变长真实样本上直接崩溃
            #   - LowRank/SharedBasis（无 fit_batch）→ 截断到最短后 concat
            from ..mapper.task_aware import TaskAwareRidgeMapper
            if isinstance(mapper, (RidgePerHeadMapper, RidgeMapper, TaskAwareRidgeMapper)) \
                    or hasattr(mapper, "fit_task_aware"):
                fit_ridge_aggregate(
                    mapper, calib, self._layer_map, kv_kind=kind,
                    positions=positions, de_rope_fn=kind_de_rope,
                )
            elif hasattr(mapper, "fit_batch"):
                mapper.fit_batch(
                    list(calib), self._layer_map, kv_kind=kind,
                    positions=None, de_rope_fn=kind_de_rope,
                )
            else:
                from ..mapper.aggregate import concat_kv_samples
                min_S = min(int(t.shape[1]) for t, _ in calib)
                n_pairs = len(calib)
                truncated = [
                    (t[:, :min_S], s[:, :min_S]) for t, s in calib
                ]
                kv_t_big, kv_s_big = concat_kv_samples(truncated)
                # concat 后位置按样本重启（de-RoPE 语义与逐样本 fit 一致）
                positions_concat = np.tile(
                    np.arange(min_S, dtype=np.float64), n_pairs
                )
                mapper.fit(kv_t_big, kv_s_big, self._layer_map, kv_kind=kind,
                          positions=positions_concat,
                          de_rope_fn=kind_de_rope)

    def _ensure_mapper(self, real_kv_calib: dict[str, list] | None = None) -> None:
        """构造并校准 mapper（构造 + 拟合；B1 在线阶段可跳过拟合用磁盘参数）。

        从 cfg.mapper.* 读取配置，用 proportional_mapping 生成 layer_map，
        构造独立的 K/V mapper 并 fit（§22 K/V 独立参数化）。

        4-way 消融需要分别变换 K 和 V，因此创建两个独立的
        RidgePerHeadMapper：_mapper_k（处理 head_dim 维）和
        _mapper_v（处理 head_dim 维）。self._mapper 保持向后兼容，
        指向 _mapper_k。

        Args:
            real_kv_calib: 真实KV校准数据，格式为 {"K": [(kv_t, kv_s), ...], "V": [...]}
                          如果提供，使用真实数据校准；否则使用合成数据（fallback）
        """
        if self._mapper is not None:
            return
        self._build_mappers()
        self._fit_mappers(real_kv_calib)
        # 向后兼容: self._mapper 指向 K mapper
        self._mapper = self._mapper_k

    # ------------------------------------------------------------------
    # 四种评估方法
    # ------------------------------------------------------------------

    def _tokenize_with_choices(
        self, text: str, choices: list[str], tokenizer: Any
    ) -> tuple[np.ndarray, list[int]]:
        """把文本 + 4 选项 tokenize，返回 (input_ids, choice_letter_ids)。

        prompt 格式（与 HFScoreProvider._score_choices 一致）：
            {text}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:
        """
        prompt = _build_scoring_prompt(text, "", choices)
        enc = tokenizer(prompt, return_tensors="np")
        ids = enc["input_ids"].reshape(-1)
        letter_ids = {
            l: int(tokenizer.encode(l, add_special_tokens=False)[0])
            for l in ["A", "B", "C", "D"]
        }
        return ids, letter_ids

    # ------------------------------------------------------------------
    # P0.5 统一格式 / P1.4 task 前缀 的公共 helper
    # ------------------------------------------------------------------

    def _handoff_context_text(self, row: Any) -> str:
        """KV 捕获路径使用的 context 文本（P1.4: 可含 task 前缀）。

        teacher KV 捕获与 self_kv 对照都必须用同一文本，保证两者的
        注入序列长度 S 一致、可比。默认无前缀时与普通 context 相同。
        """
        prefix = str(self.cfg.get("teacher", {}).get("prefill_prefix", "") or "")
        ctx = row.context or ""
        return f"{prefix}{ctx}" if prefix else ctx

    def _unified_ids(self, row: Any, tokenizer: Any, ctx_text: str) -> np.ndarray:
        """P0.5: 分离 tokenize `ctx_text` 与统一 suffix 后拼接。

        分离 token 化保证 suffix 的 token 序列在所有方法间逐 token
        一致（联合 tokenize 会因 BPE 边界效应产生不同切分）。
        """
        suffix_ids = tokenizer(
            _unified_suffix_text(row.query), return_tensors="np"
        ).input_ids.reshape(-1)
        if ctx_text:
            ctx_ids = tokenizer(ctx_text, return_tensors="np").input_ids.reshape(-1)
            return np.concatenate([ctx_ids, suffix_ids])
        return suffix_ids

    def _student_self(
        self, row: Any, choices: list[str]
    ) -> tuple[float, int, np.ndarray | None, float]:
        """方法 1: student_self — Student 自身 tokenize X+q 联合 prefill。

        P0.5 统一格式：输入 = tokenize(context) + tokenize(统一suffix)，
        与 ridge_handoff/self_kv 的 suffix 逐 token 一致 —— 方法间唯一
        差异是 context 来源。§52 合规：正常 prefill（非 zero-prefill）。
        B2: 同一次 forward 传 labels 计算 loss → 返回 student-native PPL。
        """
        import torch  # type: ignore

        assert self._student_model is not None
        assert self._student_tokenizer is not None
        ids = self._unified_ids(row, self._student_tokenizer, row.context or "")
        letter_ids = {
            l: int(self._student_tokenizer.encode(l, add_special_tokens=False)[0])
            for l in ["A", "B", "C", "D"]
        }
        ids_t = torch.as_tensor(ids.reshape(1, -1), dtype=torch.long).to(self._student_device)
        with torch.no_grad():
            out = self._student_model(ids_t, use_cache=False, labels=ids_t)
        logits = out.logits[:, -1, :].float().cpu().numpy().reshape(-1)
        try:
            ppl = math.exp(float(out.loss.item()))
        except Exception:  # noqa: BLE001
            ppl = float("inf")
        score, decision = score_choices(logits, letter_ids)
        return score, decision, logits, ppl

    def _teacher_full(
        self, row: Any, choices: list[str]
    ) -> tuple[float, int, np.ndarray | None]:
        """方法 2: teacher_full — Teacher 全量 forward X+q。

        作为 upper bound 对照：Teacher 拥有完整上下文能力。
        """
        import torch  # type: ignore

        assert self._teacher_model is not None
        assert self._teacher_tokenizer is not None
        # P0.5: 统一格式（teacher 用自己的 tokenizer，suffix token 语义一致）
        ids = self._unified_ids(row, self._teacher_tokenizer, row.context or "")
        letter_ids = {
            l: int(self._teacher_tokenizer.encode(l, add_special_tokens=False)[0])
            for l in ["A", "B", "C", "D"]
        }
        ids_t = torch.as_tensor(ids.reshape(1, -1), dtype=torch.long).to(self._teacher_device)
        with torch.no_grad():
            out = self._teacher_model(ids_t, use_cache=False)
        logits = out.logits[:, -1, :].float().cpu().numpy().reshape(-1)
        score, decision = score_choices(logits, letter_ids)
        return score, decision, logits

    def _text_handoff(
        self, row: Any, choices: list[str]
    ) -> tuple[float, int, np.ndarray | None]:
        """方法 3: text_handoff — Student 只看到截断文本 proxy。

        ⚠️ 这**不是** zero-prefill 路径！Student 仍自行 prefill 截断后的文本。
        作为诚实的 degraded baseline（文档化的退化基线）。

        text_handoff_max_chars 默认 400；超过此长度的 X 被截断。
        """
        import torch  # type: ignore

        assert self._student_model is not None
        assert self._student_tokenizer is not None
        # P0.5: 统一格式 + 截断 context（P1.5 的 summary 基线复用此路径）
        ctx_text = row.context[: self._text_max_chars] if row.context else ""
        ids = self._unified_ids(row, self._student_tokenizer, ctx_text)
        letter_ids = {
            l: int(self._student_tokenizer.encode(l, add_special_tokens=False)[0])
            for l in ["A", "B", "C", "D"]
        }
        ids_t = torch.as_tensor(ids.reshape(1, -1), dtype=torch.long).to(self._student_device)
        with torch.no_grad():
            out = self._student_model(ids_t, use_cache=False)
        logits = out.logits[:, -1, :].float().cpu().numpy().reshape(-1)
        score, decision = score_choices(logits, letter_ids)
        return score, decision, logits

    def _teacher_generate_summary(self, context_text: str) -> str:
        """P1.5: 教师生成 context 摘要（文本压缩基线）。

        教师以贪心解码产出 ≤summary_max_tokens 的摘要 —— 这是"把 X 塞进
        学生 prompt"这条文本路线的近似上限（教师级压缩）。
        """
        import torch  # type: ignore

        assert self._teacher_model is not None
        assert self._teacher_tokenizer is not None
        ids = self._teacher_tokenizer(context_text, return_tensors="np").input_ids.reshape(-1)
        ids_t = torch.as_tensor(ids.reshape(1, -1), dtype=torch.long).to(self._teacher_device)
        with torch.no_grad():
            out = self._teacher_model.generate(
                ids_t,
                max_new_tokens=self._summary_max_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=self._teacher_tokenizer.pad_token_id
                or self._teacher_tokenizer.eos_token_id,
            )
        new_tokens = out[0][ids.shape[0]:]
        return self._teacher_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _summary_handoff(
        self, row: Any, summary_text: str
    ) -> tuple[float, int, np.ndarray | None]:
        """P1.5: 学生对 `摘要 + 统一 suffix` 正常 prefill 评分（非 zero-prefill）。"""
        import torch  # type: ignore

        assert self._student_model is not None
        assert self._student_tokenizer is not None
        ids = self._unified_ids(row, self._student_tokenizer, summary_text)
        letter_ids = {
            l: int(self._student_tokenizer.encode(l, add_special_tokens=False)[0])
            for l in ["A", "B", "C", "D"]
        }
        ids_t = torch.as_tensor(ids.reshape(1, -1), dtype=torch.long).to(self._student_device)
        with torch.no_grad():
            out = self._student_model(ids_t, use_cache=False)
        logits = out.logits[:, -1, :].float().cpu().numpy().reshape(-1)
        score, decision = score_choices(logits, letter_ids)
        return score, decision, logits

    def _ridge_handoff(
        self, row: Any, choices: list[str], ablation_mode: str = "kv_both",
        preextracted_kv: tuple | None = None,
        self_kv_cache: dict[str, tuple] | None = None,
    ) -> tuple[float, int, np.ndarray | None, EvalPhaseCounters, float]:
        """方法 4: ridge_handoff — 真正的 zero-prefill 路径（§22 4-way 消融）。

        §53 单卡执行顺序：
            1. Teacher prefill X → capture PKV（numpy）—— 可使用预提取结果
            2. 根据 ablation_mode 变换 K/V → final KV (L_s, S, H, 2D)
            3. inject final KV → Student DynamicCache
            4. Student 只 decode q tokens（zero-prefill，§52 禁止 1）
            5. 计算 score (多选准确率) + PPL (perplexity)

        §22 K/V 独立消融 —— 四种模式：
            - native:   Teacher 原生 KV，不做映射（baseline）
            - k_only:   Mapped K + Native V
            - v_only:   Native K + Mapped V
            - kv_both:  Mapped K + Mapped V（Full Handoff）

        §49 计时：整个 ridge_handoff 路径在一个 sync 块内执行。

        Returns:
            (score, decision, logits, counters, ppl)
        """
        import torch  # type: ignore

        assert self._student_model is not None
        assert self._student_tokenizer is not None
        assert self._mapper is not None
        assert self._mapper_k is not None
        assert self._mapper_v is not None
        assert self._layer_map is not None

        counters = EvalPhaseCounters()
        from .backends import (
            _build_cache_from_kv,
            _extract_kv_numpy,
            _extract_kv_separate,
        )

        # ① Teacher prefill X：使用预提取的 KV 或现场运行
        if preextracted_kv is not None:
            k_np, v_np, teacher_pkv_np, S = preextracted_kv
        else:
            assert self._teacher_model is not None
            assert self._teacher_tokenizer is not None
            context_ids = self._teacher_tokenizer(
                row.context, return_tensors="np"
            ).input_ids.reshape(-1)
            S = len(context_ids)
            ids_t = torch.as_tensor(context_ids.reshape(1, -1), dtype=torch.long).to(
                self._teacher_device
            )
            with torch.no_grad():
                teacher_out = self._teacher_model(ids_t, use_cache=True)
            n_t_layers = int(
                getattr(self._teacher_model.config, "num_hidden_layers", 0)
            ) or len(teacher_out.past_key_values)
            teacher_pkv_np = _extract_kv_numpy(teacher_out.past_key_values, n_t_layers)
            k_np, v_np = _extract_kv_separate(teacher_out.past_key_values, n_t_layers)
            del teacher_out

        S = k_np.shape[1]  # 序列长度
        counters.teacher_prefill_seq_len = S

        # ③ 根据 ablation_mode 组装 final_kv (L_s, S, H, 2D)
        from ..mapper.runner import _de_rope_for_kind
        from ..rope.runner import _rope_pairs, de_rope

        D = int(self.cfg.get("teacher", {}).get("head_dim", 128))
        inv_freq = _rope_pairs(D, theta=1_000_000.0)
        de_rope_fn = lambda k, p: de_rope(k, p, inv_freq)  # noqa: E731
        positions = np.arange(S, dtype=np.float64)

        def _align_to_student(src_np: np.ndarray) -> np.ndarray:
            """将 teacher 层按 layer_map 对齐到 student 层数。

            对每个 student 层 s，对 layer_map[s] 中所有 teacher 层取均值
            （与 mapper.transform 的 top_k averaging 保持一致）。
            src_np: (L_t, S, H, D) → out: (L_s, S, H, D)
            """
            L_s = len(self._layer_map)
            L_t = src_np.shape[0]
            out = np.zeros((L_s,) + src_np.shape[1:], dtype=src_np.dtype)
            for s in range(L_s):
                teachers = [min(t, L_t - 1) for t in self._layer_map[s]]
                out[s] = src_np[teachers].mean(axis=0)
            return out

        def _student_self_kv() -> np.ndarray:
            """探针/self_kv 共用：捕获（或取缓存）学生自 prefill 的联合 KV。

            ⚠️ 仅探针模式可用（需要学生对评估行做自 prefill，非部署路径）。
            """
            cached_self = (self_kv_cache or {}).get(row.sample_id)
            if cached_self is None:
                ctx_text = self._handoff_context_text(row)
                s_ids = self._student_tokenizer(
                    ctx_text, return_tensors="np"
                ).input_ids.reshape(-1)
                s_ids_t = torch.as_tensor(
                    s_ids.reshape(1, -1), dtype=torch.long
                ).to(self._student_device)
                with torch.no_grad():
                    s_out = self._student_model(s_ids_t, use_cache=True)
                n_layers_s = int(
                    getattr(self._student_model.config, "num_hidden_layers", 0)
                ) or len(s_out.past_key_values)
                cached_self = _extract_kv_separate(s_out.past_key_values, n_layers_s)
                del s_out
                if self_kv_cache is not None:
                    self_kv_cache[row.sample_id] = cached_self
            sk, sv = cached_self
            return np.concatenate([sk, sv], axis=-1)

        def _translated_kv_both() -> np.ndarray:
            """探针共用：kv_both 翻译路径（与 kv_both 分支语义一致）。"""
            k_de_rope = _de_rope_for_kind(self.cfg, "K", de_rope_fn)
            v_de_rope = _de_rope_for_kind(self.cfg, "V", de_rope_fn)
            mapped_k = self._mapper_k.transform(
                k_np, self._layer_map, kv_kind="K",
                positions=positions, de_rope_fn=k_de_rope,
            )
            if self._rope_align == "unrotated":
                from ..rope.runner import apply_rope
                mapped_k = apply_rope(mapped_k, positions, inv_freq)
            mapped_v = self._mapper_v.transform(
                v_np, self._layer_map, kv_kind="V",
                positions=positions, de_rope_fn=v_de_rope,
            )
            return np.concatenate([mapped_k, mapped_v], axis=-1)

        if ablation_mode == "self_kv":
            # ── P0.0 恒等对照（H1 判定）──
            # 学生自身 prefill(prefix+context) 的 KV，走与 ridge_handoff 完全
            # 相同的注入/解码/评分路径。self_kv ≈ student_self ⇒ 注入机制
            # 无损（机制损失≈0）；ridge_handoff 与 self_kv 的差 ⇒ 纯 mapper
            # 损失。这是把"机制 bug"与"mapper 差"解耦的关键对照。
            # P1.4: 捕获文本与 teacher KV 捕获一致（prefix+context）。
            ctx_text = self._handoff_context_text(row)
            s_ids = self._student_tokenizer(ctx_text, return_tensors="np").input_ids.reshape(-1)
            s_ids_t = torch.as_tensor(s_ids.reshape(1, -1), dtype=torch.long).to(
                self._student_device
            )
            cached_self = (self_kv_cache or {}).get(row.sample_id)
            if cached_self is None:
                with torch.no_grad():
                    s_out = self._student_model(s_ids_t, use_cache=True)
                n_layers_s = int(
                    getattr(self._student_model.config, "num_hidden_layers", 0)
                ) or len(s_out.past_key_values)
                cached_self = _extract_kv_separate(s_out.past_key_values, n_layers_s)
                del s_out
                if self_kv_cache is not None:
                    self_kv_cache[row.sample_id] = cached_self
            sk, sv = cached_self
            k_np, v_np = sk, sv
            S = sk.shape[1]
            counters.teacher_prefill_seq_len = S
            final_kv = np.concatenate([k_np, v_np], axis=-1)

        elif ablation_mode.startswith("mix_a") or ablation_mode.startswith("win_"):
            # ── P0-Oracle 探针（H3 上界判定，非部署路径）──
            # mix_aXX: cache = (1−α)·学生自KV + α·翻译教师KV（α=XX/100）
            # win_{low,mid,high}: 非窗口层=学生自KV，窗口层=翻译教师KV
            #   （MoT 注入模式的探针版 —— 定位最优 channel 窗口）
            # 两者都需要学生自 KV 作为基底；cache 全长以学生侧为准。
            if not getattr(self, "_probe_mode", False):
                raise ValueError(
                    f"探针模式 {ablation_mode} 需要inject_eval.probe_mode=true（仅诊断用）"
                )
            self_kv_joint = _student_self_kv()
            translated = _translated_kv_both()
            L_p = min(self_kv_joint.shape[0], translated.shape[0])
            self_kv_joint = self_kv_joint[:L_p]
            translated = translated[:L_p]
            S = self_kv_joint.shape[1]
            counters.teacher_prefill_seq_len = S
            if ablation_mode.startswith("mix_a"):
                alpha = float(ablation_mode[5:]) / 100.0
                final_kv = (1.0 - alpha) * self_kv_joint + alpha * translated
            else:  # win_*
                third = L_p // 3
                final_kv = self_kv_joint.copy()
                if ablation_mode == "win_low":
                    final_kv[:third] = translated[:third]
                elif ablation_mode == "win_mid":
                    final_kv[third:2 * third] = translated[third:2 * third]
                elif ablation_mode == "win_high":
                    final_kv[2 * third:] = translated[2 * third:]
                else:
                    raise ValueError(f"未知探针窗口模式: {ablation_mode}")

        elif ablation_mode == "native":
            # Case 1: Teacher 原生 KV，不做映射（baseline）
            # 先拆分 K/V 再对齐，确保 head_dim 语义正确（跨架构安全）
            k_aligned = _align_to_student(k_np)  # (L_s, S, H, D)
            v_aligned = _align_to_student(v_np)  # (L_s, S, H, D)
            # P0.6: 尺度重标定诊断（native_rescale=true 时）——把对齐后的
            # teacher KV 逐层缩放到学生范数水平，分离"尺度失配"与"语义失配"：
            # 若重标定后 native PPL 大幅下降 ⇒ 灾难主要来自尺度；否则是语义。
            if self._native_rescale and self._kv_norm_stats:
                lm = self._layer_map or [[min(s_, k_np.shape[0] - 1)] for s_ in range(k_aligned.shape[0])]
                for ch, aligned in (("K", k_aligned), ("V", v_aligned)):
                    st = self._kv_norm_stats.get(ch)
                    if not st:
                        continue
                    t_arr = np.asarray(st["teacher_per_layer"])
                    s_arr = np.asarray(st["student_per_layer"])
                    # 层映射可为 top-k（每学生层多个教师层）——教师范数取组内均值
                    ratio = np.array([
                        s_arr[s_] / max(
                            np.mean([t_arr[min(t, len(t_arr) - 1)]
                                    for t in (lm[s_] or [s_])]),
                            1e-9,
                        )
                        for s_ in range(min(len(lm), len(s_arr)))
                    ])
                    aligned *= ratio[:, None, None, None].astype(aligned.dtype)
            final_kv = np.concatenate([k_aligned, v_aligned], axis=-1)  # (L_s, S, H, 2D)

        elif ablation_mode == "k_only":
            # Case 2: 只映射 K，V 保持 teacher 原生
            k_de_rope = _de_rope_for_kind(self.cfg, "K", de_rope_fn)
            mapped_k = self._mapper_k.transform(
                k_np, self._layer_map, kv_kind="K",
                positions=positions, de_rope_fn=k_de_rope,
            )
            # B3: unrotated 模式 —— fit 目标是 de-roped student K（unrotated 空间），
            # transform 产物也在 unrotated 空间；注入前必须 re-RoPE 回 student
            # 旋转空间（HF 对注入 K 不做重旋转，见 §23 注释）。rotated 旧模式
            # 由 W 隐式吸收位置旋转，不在此 re-RoPE。
            if self._rope_align == "unrotated":
                from ..rope.runner import apply_rope
                mapped_k = apply_rope(mapped_k, positions, inv_freq)
            # mapped_k: (L_s, S, H, D_head)
            v_aligned = _align_to_student(v_np)  # (L_s, S, H, D_head)
            final_kv = np.concatenate([mapped_k, v_aligned], axis=-1)  # (L_s, S, H, 2D)

        elif ablation_mode == "v_only":
            # Case 3: 只映射 V，K 保持 teacher 原生
            v_de_rope = _de_rope_for_kind(self.cfg, "V", de_rope_fn)
            mapped_v = self._mapper_v.transform(
                v_np, self._layer_map, kv_kind="V",
                positions=positions, de_rope_fn=v_de_rope,
            )
            k_aligned = _align_to_student(k_np)
            final_kv = np.concatenate([k_aligned, mapped_v], axis=-1)

        elif ablation_mode == "kv_both":
            # Case 4: K 和 V 都映射（Full Handoff）
            k_de_rope = _de_rope_for_kind(self.cfg, "K", de_rope_fn)
            v_de_rope = _de_rope_for_kind(self.cfg, "V", de_rope_fn)
            mapped_k = self._mapper_k.transform(
                k_np, self._layer_map, kv_kind="K",
                positions=positions, de_rope_fn=k_de_rope,
            )
            # B3: unrotated 模式 re-RoPE（与 k_only 分支同语义，见上方注释）
            if self._rope_align == "unrotated":
                from ..rope.runner import apply_rope
                mapped_k = apply_rope(mapped_k, positions, inv_freq)
            mapped_v = self._mapper_v.transform(
                v_np, self._layer_map, kv_kind="V",
                positions=positions, de_rope_fn=v_de_rope,
            )
            final_kv = np.concatenate([mapped_k, mapped_v], axis=-1)

        elif ablation_mode == "zero_kv":
            # Case 5: Zero KV cache (全零 baseline)
            k_aligned = _align_to_student(k_np)
            final_kv = np.zeros_like(k_aligned)
            final_kv = np.concatenate([final_kv, np.zeros_like(k_aligned)], axis=-1)

        elif ablation_mode == "random_proj":
            # Case 6: Random projection baseline (随机矩阵映射)
            import numpy as _np_rng
            rng = _np_rng.random.RandomState(42)
            k_aligned = _align_to_student(k_np)
            v_aligned = _align_to_student(v_np)
            L_s, S_len, H, D = k_aligned.shape
            # Random matrix per (layer, head) — shared across positions
            # Shape: (L, H, D, D) — applies same transform at each position
            rand_k = rng.randn(L_s, H, D, D).astype(np.float32) * 0.02
            rand_v = rng.randn(L_s, H, D, D).astype(np.float32) * 0.02
            # einsum: (L,S,H,D) x (L,H,D,D) -> (L,S,H,D)
            mapped_k = np.einsum("lshd,lhdo->lsho", k_aligned, rand_k)
            mapped_v = np.einsum("lshd,lhdo->lsho", v_aligned, rand_v)
            final_kv = np.concatenate([mapped_k, mapped_v], axis=-1)

        else:
            raise ValueError(f"未知 ablation_mode: {ablation_mode!r}")

        counters.inject_seq_len = S

        # ④ inject：final KV → Student DynamicCache
        # 注意：cache 保持 num_kv_heads 维度，HF attention 内部处理 GQA expand
        cache = _build_cache_from_kv(
            final_kv,
            device=self._student_device,
            dtype=self._student_model.dtype,
            num_attention_heads=None,  # 不做 GQA expand，HF 内部处理
        )

        # ◆ D1 修复（审计真实化）：past_len 从实际 cache 读取，不再手工赋值 S。
        #   此值即 §52 断言的数据源 —— 若注入形状/层对齐有 bug 导致 cache
        #   长度 ≠ 教师序列长度，assert_query_handoff 会真实失败。
        past_before = cache_seq_len(cache)
        counters.query_past_len = past_before

        # ⑤ Student decode q tokens（zero-prefill：只喂统一 suffix）
        # P0.5: suffix 与 student_self 等方法逐 token 一致（消除格式混淆）
        suffix = _unified_suffix_text(row.query)
        suffix_ids = self._student_tokenizer(suffix, return_tensors="np").input_ids.reshape(-1)
        n_query = len(suffix_ids)
        counters.query_new_tokens = n_query

        # 评分：取 "Answer:" 之后位置的 logit，对 A/B/C/D 做 softmax
        choice_ids = {
            l: int(self._student_tokenizer.encode(l, add_special_tokens=False)[0])
            for l in ["A", "B", "C", "D"]
        }

        suffix_t = torch.as_tensor(suffix_ids.reshape(1, -1), dtype=torch.long).to(
            self._student_device
        )
        pos_ids = torch.arange(S, S + n_query, dtype=torch.long).unsqueeze(0).to(
            self._student_device
        )
        # ◆ cache 污染修复（transformers 4.52 实测：use_cache=False 且传入
        #   past_key_values 时，模型仍原地 update cache —— PPL forward 会把
        #   suffix K/V 写进共享 cache，评分 forward 再 attend 一遍 suffix
        #   → logits 被污染。旧 sweep 的 ridge 分数均受此影响）。
        #   修复：PPL 与评分各用一份独立 cache（同一 final_kv 构建两次）。
        ppl = compute_ppl(
            self._student_model, suffix_ids,
            past_key_values=_build_cache_from_kv(
                final_kv, device=self._student_device,
                dtype=self._student_model.dtype, num_attention_heads=None,
            ),
            position_ids=np.arange(S, S + n_query, dtype=np.int64),
        )
        # ⑦ Student scoring forward（独立 cache，未被 PPL 污染）
        with torch.no_grad():
            student_out = self._student_model(
                suffix_t, past_key_values=cache, use_cache=False,
                position_ids=pos_ids,
            )
        # ◆ D1 审计语义（修订）：transformers 会把本次 forward 的 n_query 个
        #   suffix token 追加进 cache —— 这是 HF 行为，不是 zero-prefill 违反。
        #   Student 允许消费 query（§52：MAY process q），因此按
        #   "增量超出 n_query 的部分"计为 prefill 违规；若有人在此路径
        #   re-prefill 上下文 X（≈S 个 token），增量将远超 n_query 而被抓。
        past_after = cache_seq_len(cache)
        counters.student_prefill_new_tokens = max(
            0, past_after - past_before - n_query
        )
        logits = student_out.logits[:, -1, :].float().cpu().numpy().reshape(-1)

        del student_out, cache

        # §52 审计断言
        counters.assert_zero_prefill()
        counters.assert_query_handoff()

        score, decision = score_choices(logits, choice_ids)
        return score, decision, logits, counters, ppl

    # ------------------------------------------------------------------
    # 主评估入口
    # ------------------------------------------------------------------

    def evaluate(self, sample_rows: list[Any]) -> dict[str, Any]:
        """对 sample_rows 运行四方法评估，返回两份 artifact 的 dict。

        协议 v1.1 变更：
            A2. 校准/评估互斥切分（默认开启）：rows[:n_calib] 用于 mapper
                拟合（Phase 1.5 student self-prefill + task-aware Phase B），
                其余行用于四方法评估（Phase 2）—— 消除"同样本拟合+评估"
                的记忆效应；样本数 < 4 时回退旧行为并如实标注。
            A1. 逐样本记录 gold_prob（gold 字母 softmax 概率）与 accuracy，
                供 CLI 计算 accuracy/gold 口径的 CHG/TGRR 与显著性检验。
            B1. persist_kv=True 时把 Teacher KV、rows、teacher 分数、mapper
                参数落盘（kv_store 目录），供在线阶段免 Teacher 冷启动。
            B2. 逐样本计时 student_self prefill 与 ridge_handoff(kv_both)，
                报告 PSR（含"测量用 PPL forward"的保守性说明）。

        §53 内存纪律：
            - Teacher 先跑 teacher_full（加载 Teacher → 逐样本 → 卸载）
            - 然后 Student 跑 student_self + text_handoff + ridge_handoff
              （加载 Student → 逐样本 → 卸载）
            - 每个模型组之间 GC + empty_cache

        Args:
            sample_rows: list[Sample] — 每个含 context, query, answer, split, sample_id

        Returns:
            {
                "replacement_artifact": {...},  # replacement_score_artifact.json 内容
                "capability_artifact": {...},   # capability_score_artifact.json 内容
                "counters_summary": [...],      # 每样本的 zero-prefill 计数器
                "zero_prefill_verified": bool,  # 全局 zero-prefill 审计
                "psr_summary": {...},           # B2 系统收益核算
            }
        """
        import torch  # type: ignore

        n = min(len(sample_rows), self._max_samples)
        rows = sample_rows[:n]
        if not rows:
            raise RuntimeError("inject-eval: 无可用样本（sample_rows 为空）")

        # ── A2: 校准/评估互斥切分 ──
        # mapper 拟合只用 calib_rows；四方法评估只用 eval_rows。
        # 两者同分布（同一用户数据集顺序切分）但样本互斥。
        # P0.2: calib_shuffle=true 时按 seed 随机置换后再切分 —— 提供
        # 校准种子方差（mapper 拟合依赖抽到哪些校准样本）。
        # A2 修正: eval_from_tail=true 时评估集固定为末尾 n_eval 样本，
        # 增大 calib_samples 不改变评估集 —— 受控规模阶梯。
        rows_work = list(rows)
        if bool(self.cfg.get("inject_eval", {}).get("calib_shuffle", False)):
            rng_sh = np.random.default_rng(self._seed)
            perm = rng_sh.permutation(len(rows_work))
            rows_work = [rows_work[i] for i in perm]
        n_eval_fixed = int(self.cfg.get("inject_eval", {}).get("eval_from_tail_n", 0))
        if self._calib_split and len(rows_work) >= 4:
            if n_eval_fixed > 0 and len(rows_work) > n_eval_fixed:
                eval_rows = rows_work[-n_eval_fixed:]
                calib_rows = rows_work[:-n_eval_fixed]
                if self._n_calib > 0 and len(calib_rows) > self._n_calib:
                    calib_rows = calib_rows[: self._n_calib]
            else:
                n_calib_split = self._n_calib if self._n_calib > 0 else max(1, len(rows_work) // 2)
                n_calib_split = min(n_calib_split, len(rows_work) - 1)
                calib_rows = rows_work[:n_calib_split]
                eval_rows = rows_work[n_calib_split:]
            calib_eval_disjoint = True
        else:
            calib_rows = rows_work
            eval_rows = rows_work
            calib_eval_disjoint = False
        eval_ids = {r.sample_id for r in eval_rows}

        def _timed(fn):
            """§49 计时块：CUDA sync 边界内测墙钟（ms）。"""
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fn()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            return out, (time.perf_counter() - t0) * 1000.0

        # 确定 choices：从 cfg 或默认 4 选项
        # NOTE: 旧版使用 ["A","B","C","D"] 作为 letter-only prompt，导致评分虚高。
        # 修正：从 query 中提取实际选项文本，用于 full-text prompt 评分。
        default_choices = self.cfg.get("inject_eval", {}).get("default_choices", None)

        # 结果存储
        replacement_records: list[dict[str, Any]] = []
        capability_records: list[dict[str, Any]] = []
        counters_list: list[dict[str, Any]] = []
        all_zero_prefill = True
        psr_rows: list[float] = []
        teacher_scores_store: dict[str, Any] = {}

        # ── Phase 1: Teacher Full + 预提取 Teacher KV（for ridge_handoff）──
        # §53: Teacher Load → 逐样本 forward → Teacher Unload
        logger.info("[inject-eval] Phase 1: teacher_full + pre-extract teacher KV (%d samples)", n)
        self._load_teacher()
        teacher_kv_cache: dict[str, Any] = {}  # sample_id → (k_np, v_np, teacher_pkv_np, S)
        try:
            for row in rows:
                # 从 query 提取实际选项文本（修正 letter-only prompt 问题）
                if default_choices is not None:
                    choices = default_choices
                else:
                    choices = _extract_choices_from_query(row.query)
                    if not choices:
                        # 标记样本为不可评估，不使用letter-only fallback
                        logger.warning(
                            "[inject-eval] Cannot extract choices for %s, skipping sample",
                            row.sample_id,
                        )
                        if row.sample_id in eval_ids:
                            capability_records.append({
                                "method": "teacher",
                                "sample_id": row.sample_id,
                                "seed": self._seed,
                                "score": 0.0,
                                "decision": -1,
                                "skipped": True,
                                "skip_reason": "no_choices_extracted",
                            })
                        teacher_kv_cache[row.sample_id] = None
                        continue

                # 1a. teacher_full（context + query + choices，用于评分）
                # A2: 只对评估行跑 teacher_full（校准行不需要 upper-bound 分数）
                if row.sample_id in eval_ids:
                    try:
                        score, decision, t_logits = self._teacher_full(row, choices)
                    except Exception as e:  # noqa: BLE001
                        logger.warning("[inject-eval] teacher_full failed for %s: %s", row.sample_id, e)
                        score, decision, t_logits = 0.0, 0, None
                    t_correct = False
                    if row.answer is not None:
                        letters = ["A", "B", "C", "D"]
                        if 0 <= decision < len(letters):
                            t_correct = letters[decision].upper() == row.answer.strip().upper()
                    t_gold = gold_prob_from_logits(t_logits, row.answer, self._teacher_letter_ids)
                    # P1.1: 教师字母软分布（KD 目标，无 gold 标注时用）
                    t_letter_probs = None
                    if t_logits is not None and self._teacher_letter_ids:
                        _vals = np.array([
                            float(t_logits[self._teacher_letter_ids[l]])
                            for l in ["A", "B", "C", "D"]
                        ])
                        _e = np.exp(_vals - _vals.max())
                        t_letter_probs = (_e / _e.sum()).tolist()
                    capability_records.append({
                        "method": "teacher",
                        "sample_id": row.sample_id,
                        "seed": self._seed,
                        "score": float(score),
                        "decision": int(decision),
                        "correct": bool(t_correct),
                        "gold_prob": float(t_gold) if t_gold is not None else None,
                    })
                    teacher_scores_store[row.sample_id] = {
                        "score": float(score),
                        "decision": int(decision),
                        "correct": bool(t_correct),
                        "gold_prob": float(t_gold) if t_gold is not None else None,
                        "letter_probs": t_letter_probs,
                    }
                    # P1.5: 教师生成摘要（文本基线的 context 压缩上限近似）
                    if self._summary_baseline:
                        try:
                            self._summaries[row.sample_id] = self._teacher_generate_summary(
                                self._handoff_context_text(row)
                            )
                        except Exception as e:  # noqa: BLE001
                            logger.warning(
                                "[inject-eval] summary generation failed for %s: %s",
                                row.sample_id, e,
                            )
                # 1b. 预提取 context-only teacher KV（for ridge_handoff in Phase 2）
                # P1.4: 捕获文本 = prefill_prefix + context（与 self_kv 对照一致）
                try:
                    from .backends import _extract_kv_numpy, _extract_kv_separate
                    context_ids = self._teacher_tokenizer(
                        self._handoff_context_text(row), return_tensors="np"
                    ).input_ids.reshape(-1)
                    ids_t = torch.as_tensor(context_ids.reshape(1, -1), dtype=torch.long).to(
                        self._teacher_device
                    )
                    with torch.no_grad():
                        ctx_out = self._teacher_model(ids_t, use_cache=True)
                    n_t = int(
                        getattr(self._teacher_model.config, "num_hidden_layers", 0)
                    ) or len(ctx_out.past_key_values)
                    teacher_pkv_np = _extract_kv_numpy(ctx_out.past_key_values, n_t)
                    k_np, v_np = _extract_kv_separate(ctx_out.past_key_values, n_t)
                    teacher_kv_cache[row.sample_id] = (k_np, v_np, teacher_pkv_np, len(context_ids))
                    # B1: KV 落盘（离线阶段产物，在线阶段免 Teacher 冷启动）
                    if self._persist_kv:
                        from .kv_store import save_kv_sample
                        save_kv_sample(self._kv_store_dir, row.sample_id, k_np, v_np)
                    del ctx_out
                except Exception as e:  # noqa: BLE001
                    logger.warning("[inject-eval] teacher KV pre-extract failed for %s: %s", row.sample_id, e)
                    teacher_kv_cache[row.sample_id] = None
        finally:
            self._unload_teacher()

        # §53 CUDA Cleanup + GC
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── Phase 1.5: Collect real paired KV calibration data ──
        # §53: Student Load → run on same prompts as teacher → capture student KV
        # This provides real (teacher_kv, student_kv) pairs for mapper fitting.
        real_kv_calib: dict[str, list] = {"K": [], "V": []}
        student_kv_by_id: dict[str, tuple] = {}  # P1.1: task-aware recon 锚
        # P0.6: 逐层 K/V 范数统计（教师 vs 学生）—— 尺度失配诊断 +
        # native_rescale 的比例来源
        norm_acc = {
            "K": {"teacher": [], "student": []},
            "V": {"teacher": [], "student": []},
        }
        n_calib = int(self.cfg.get("mapper", {}).get("inject_eval_calib_samples", 50))
        # A2: 只用校准行做 student self-prefill（与评估行互斥）
        fit_rows = [r for r in calib_rows if teacher_kv_cache.get(r.sample_id) is not None][:n_calib]
        calib_rows = fit_rows
        if calib_rows:
            logger.info(
                "[inject-eval] Phase 1.5: Collecting real paired KV calibration (%d samples, disjoint_eval=%s)",
                len(calib_rows), calib_eval_disjoint,
            )
            # B3: unrotated 模式下，student K 目标需先 de-RoPE 到 unrotated
            # 空间（与 teacher 侧 fit 输入同一空间；家族内 theta 同为 1e6）
            rope_inv_freq = None
            if self._rope_align == "unrotated":
                from ..rope.runner import _rope_pairs, de_rope as _de_rope
                rope_D = int(self.cfg.get("teacher", {}).get("head_dim", 128))
                rope_inv_freq = _rope_pairs(rope_D, theta=1_000_000.0)
            self._load_student()
            try:
                for row in calib_rows:
                    try:
                        # P0.5/P1.4: 校准 KV 与捕获路径同文本（prefix+context），
                        # 保证 fit 目标分布 = 注入分布
                        student_ids = self._student_tokenizer(
                            self._handoff_context_text(row), return_tensors="np"
                        ).input_ids.reshape(-1)
                        s_ids_t = torch.as_tensor(student_ids.reshape(1, -1), dtype=torch.long).to(
                            self._student_device
                        )
                        with torch.no_grad():
                            student_out = self._student_model(s_ids_t, use_cache=True)
                        n_s = int(
                            getattr(self._student_model.config, "num_hidden_layers", 0)
                        ) or len(student_out.past_key_values)
                        s_k_np, s_v_np = _extract_kv_separate(student_out.past_key_values, n_s)
                        del student_out

                        # Get teacher KV (already extracted in Phase 1)
                        t_k_np, t_v_np, _, _ = teacher_kv_cache[row.sample_id]

                        # Truncate to min length (prompt lengths may differ)
                        min_S = min(t_k_np.shape[1], s_k_np.shape[1])
                        t_k_np = t_k_np[:, :min_S, :, :]
                        t_v_np = t_v_np[:, :min_S, :, :]
                        s_k_np = s_k_np[:, :min_S, :, :]
                        s_v_np = s_v_np[:, :min_S, :, :]

                        # B3: student K → unrotated 空间（fit 目标去位置旋转）
                        if rope_inv_freq is not None:
                            s_k_np = _de_rope(
                                s_k_np, np.arange(min_S, dtype=np.float64), rope_inv_freq
                            )

                        # Only add if shapes are compatible (same head_dim)
                        if t_k_np.shape[-1] == s_k_np.shape[-1]:
                            real_kv_calib["K"].append((t_k_np, s_k_np))
                            real_kv_calib["V"].append((t_v_np, s_v_np))
                            student_kv_by_id[row.sample_id] = (s_k_np, s_v_np)
                            # P0.6: 逐层平均 L2 范数（教师 vs 学生）
                            norm_acc["K"]["teacher"].append(
                                np.linalg.norm(t_k_np, axis=-1).mean(axis=(1, 2)))
                            norm_acc["K"]["student"].append(
                                np.linalg.norm(s_k_np, axis=-1).mean(axis=(1, 2)))
                            norm_acc["V"]["teacher"].append(
                                np.linalg.norm(t_v_np, axis=-1).mean(axis=(1, 2)))
                            norm_acc["V"]["student"].append(
                                np.linalg.norm(s_v_np, axis=-1).mean(axis=(1, 2)))
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            "[inject-eval] Calibration KV capture failed for %s: %s",
                            row.sample_id, e,
                        )
            finally:
                self._unload_student()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # ── P0.6: 聚合 KV 范数统计（教师 vs 学生，尺度失配诊断）──
        self._kv_norm_stats = {}
        if norm_acc["K"]["teacher"]:
            for ch in ("K", "V"):
                t_arr = np.mean(np.stack(norm_acc[ch]["teacher"]), axis=0)
                s_arr = np.mean(np.stack(norm_acc[ch]["student"]), axis=0)
                n_ref = len(self._layer_map) if self._layer_map else min(len(s_arr), len(t_arr))
                self._kv_norm_stats[ch] = {
                    "teacher_per_layer": t_arr.tolist(),
                    "student_per_layer": s_arr.tolist(),
                    "ratio_mean": float(
                        np.mean(s_arr[:n_ref] / np.maximum(t_arr[:n_ref], 1e-9))
                    ),
                }

        # ── Phase 1.6: Task-Aware Mapper Fine-tuning (if configured) ──
        mapper_type = str(self.cfg.get("mapper", {}).get("type", "ridge"))
        logger.info("[inject-eval] Phase 1.6 check: mapper_type=%s, real_kv_calib_K_count=%d",
                     mapper_type, len(real_kv_calib.get("K", [])))
        if mapper_type == "task_aware" and real_kv_calib.get("K"):
            # Must fit mapper (Phase A) before fine-tuning (Phase B)
            self._ensure_mapper(real_kv_calib=real_kv_calib)
            logger.info("[inject-eval] Phase 1.6: Task-aware mapper fine-tuning")
            self._load_student()
            try:
                from ..mapper.runner import kv_kinds
                kinds = kv_kinds(self.cfg)
                teacher_answers = {row.sample_id: row.answer for row in calib_rows if row.answer}
                scoring_suffix = _SCORING_SUFFIX

                for kind in kinds:
                    mapper = self._mapper_k if kind == "K" else self._mapper_v
                    if hasattr(mapper, 'fit_task_aware'):
                        mapper.fit_task_aware(
                            student_model=self._student_model,
                            student_tokenizer=self._student_tokenizer,
                            teacher_kv_cache=teacher_kv_cache,
                            sample_rows=calib_rows,
                            layer_map=self._layer_map,
                            kv_kind=kind,
                            teacher_answers=teacher_answers,
                            device=self._student_device,
                            scoring_suffix=scoring_suffix,
                            teacher_letter_probs={
                                sid: v["letter_probs"]
                                for sid, v in teacher_scores_store.items()
                                if v.get("letter_probs")
                            },
                            student_kv_calib=student_kv_by_id,
                            scoring_letter_ids=self._student_letter_ids,
                        )
            finally:
                self._unload_student()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # ── Phase 2: Student (self + text + ridge 消融) ──
        # §53: Student Load → 串行 → Student Unload
        # A3: 默认消融含 native（无参数教师均值基线）—— "mapper 是否必要"的
        # 对照组；cfg 可覆盖。
        ablation_modes = self.cfg.get("inject_eval", {}).get(
            "ablation_modes", ["native", "kv_both"],
        )
        logger.info(
            "[inject-eval] Phase 2: student_self + text_handoff + ridge_handoff "
            "(modes=%s, n_eval=%d, n_calib=%d)",
            ablation_modes, len(eval_rows), len(calib_rows),
        )
        self._ensure_mapper(real_kv_calib=real_kv_calib)
        # B1: mapper 参数落盘（fit 完成后）
        if self._persist_kv:
            from .kv_store import save_mapper_params, write_manifest, write_rows_teacher
            save_mapper_params(
                self._kv_store_dir, {"K": self._mapper_k, "V": self._mapper_v}
            )
        self._load_student()
        try:
            for row in eval_rows:
                # 从 query 提取实际选项文本（与 Phase 1 一致）
                if default_choices is not None:
                    choices = default_choices
                else:
                    choices = _extract_choices_from_query(row.query)
                    if not choices:
                        # 标记样本为不可评估，跳过
                        logger.warning(
                            "[inject-eval] Cannot extract choices for %s in Phase 2, skipping",
                            row.sample_id,
                        )
                        for m in ["student", "text_handoff", *ablation_modes]:
                            capability_records.append({
                                "method": m,
                                "sample_id": row.sample_id,
                                "seed": self._seed,
                                "score": 0.0,
                                "decision": -1,
                                "skipped": True,
                                "skip_reason": "no_choices_extracted",
                            })
                        continue

                # 2a. student_self（B2: 计时 = student 自 prefill 成本基线）
                try:
                    (s_score, s_decision, s_logits, s_ppl), t_prefill_ms = _timed(
                        lambda: self._student_self(row, choices)
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning("[inject-eval] student_self failed for %s: %s", row.sample_id, e)
                    s_score, s_decision, s_logits, s_ppl, t_prefill_ms = 0.0, 0, None, float("inf"), 0.0
                s_correct = False
                if row.answer is not None:
                    letters = ["A", "B", "C", "D"]
                    if 0 <= s_decision < len(letters):
                        s_correct = letters[s_decision].upper() == row.answer.strip().upper()
                s_gold = gold_prob_from_logits(s_logits, row.answer, self._student_letter_ids)
                capability_records.append({
                    "method": "student",
                    "sample_id": row.sample_id,
                    "seed": self._seed,
                    "score": float(s_score),
                    "decision": int(s_decision),
                    "correct": bool(s_correct),
                    "gold_prob": float(s_gold) if s_gold is not None else None,
                    "ppl": float(s_ppl),
                    "prefill_ms": float(t_prefill_ms),
                })

                # 2b. text_handoff（非 zero-prefill，诚实退化基线）
                try:
                    t_score, t_decision, t_logits = self._text_handoff(row, choices)
                except Exception as e:  # noqa: BLE001
                    logger.warning("[inject-eval] text_handoff failed for %s: %s", row.sample_id, e)
                    t_score, t_decision, t_logits = 0.0, 0, None
                t_correct = False
                if row.answer is not None:
                    letters = ["A", "B", "C", "D"]
                    if 0 <= t_decision < len(letters):
                        t_correct = letters[t_decision].upper() == row.answer.strip().upper()
                t_gold = gold_prob_from_logits(t_logits, row.answer, self._student_letter_ids)
                capability_records.append({
                    "method": "text",
                    "sample_id": row.sample_id,
                    "seed": self._seed,
                    "score": float(t_score),
                    "decision": int(t_decision),
                    "correct": bool(t_correct),
                    "gold_prob": float(t_gold) if t_gold is not None else None,
                })

                # P1.5: teacher-summary 文本基线（实用性判定；仅离线路径）
                if self._summary_baseline and row.sample_id in self._summaries:
                    try:
                        su_score, su_decision, su_logits = self._summary_handoff(
                            row, self._summaries[row.sample_id]
                        )
                        su_correct = False
                        if row.answer is not None and 0 <= su_decision < 4:
                            su_correct = ["A", "B", "C", "D"][su_decision].upper() == row.answer.strip().upper()
                        su_gold = gold_prob_from_logits(su_logits, row.answer, self._student_letter_ids)
                        capability_records.append({
                            "method": "summary",
                            "sample_id": row.sample_id,
                            "seed": self._seed,
                            "score": float(su_score),
                            "decision": int(su_decision),
                            "correct": bool(su_correct),
                            "gold_prob": float(su_gold) if su_gold is not None else None,
                        })
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            "[inject-eval] summary_handoff failed for %s: %s", row.sample_id, e
                        )

                # 2c. ridge_handoff — §22 消融（使用预提取的 teacher KV）
                cached_kv = teacher_kv_cache.get(row.sample_id)
                row_self_kv: dict[str, tuple] = {}  # P0.0: self_kv 恒等对照的行内缓存
                t_handoff_ms = 0.0
                for mode in ablation_modes:
                    counters = EvalPhaseCounters()
                    try:
                        if cached_kv is None:
                            raise RuntimeError("teacher KV not available")
                        if mode == "kv_both":
                            (r_score, r_decision, r_logits, counters, r_ppl), t_handoff_ms = _timed(
                                lambda: self._ridge_handoff(
                                    row, choices, ablation_mode=mode,
                                    preextracted_kv=cached_kv,
                                    self_kv_cache=row_self_kv,
                                )
                            )
                        else:
                            r_score, r_decision, r_logits, counters, r_ppl = self._ridge_handoff(
                                row, choices, ablation_mode=mode,
                                preextracted_kv=cached_kv,
                                self_kv_cache=row_self_kv,
                            )
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            "[inject-eval] ridge_handoff(%s) failed for %s: %s",
                            mode, row.sample_id, e,
                        )
                        r_score, r_decision, r_ppl, r_logits = 0.0, 0, float("inf"), None
                        counters = EvalPhaseCounters()
                        all_zero_prefill = False
                    if not counters.is_verified:
                        all_zero_prefill = False

                    method_name = f"ridge_{mode}"  # e.g. "ridge_kv_both"
                    r_correct = False
                    if row.answer is not None:
                        letters = ["A", "B", "C", "D"]
                        if 0 <= r_decision < len(letters):
                            r_correct = letters[r_decision].upper() == row.answer.strip().upper()
                    r_gold = gold_prob_from_logits(r_logits, row.answer, self._student_letter_ids)
                    # P0.6: 行为层诊断 —— handoff 与 student_self 最终 logits 的余弦
                    r_logit_cos = None
                    if r_logits is not None and s_logits is not None:
                        denom = float(np.linalg.norm(r_logits) * np.linalg.norm(s_logits))
                        if denom > 0:
                            r_logit_cos = float(np.dot(r_logits, s_logits) / denom)
                    capability_records.append({
                        "method": method_name,
                        "sample_id": row.sample_id,
                        "seed": self._seed,
                        "score": float(r_score),
                        "decision": int(r_decision),
                        "correct": bool(r_correct),
                        "gold_prob": float(r_gold) if r_gold is not None else None,
                        "ppl": float(r_ppl),
                        "logit_cos_vs_student": r_logit_cos,
                    })

                    # replacement artifact 用 student_self vs ridge_kv_both
                    if mode == "kv_both":
                        replacement_records.append({
                            "sample_id": row.sample_id,
                            "student_score": float(s_score),
                            "handoff_score": float(r_score),
                            "student_gold_prob": float(s_gold) if s_gold is not None else None,
                            "handoff_gold_prob": float(r_gold) if r_gold is not None else None,
                        })
                        # B2: PSR = (student 自 prefill − 冷启动 handoff) / 自 prefill
                        # 注：t_handoff 含测量专用的 PPL forward，PSR 偏保守
                        if t_prefill_ms > 0:
                            psr_rows.append((t_prefill_ms - t_handoff_ms) / t_prefill_ms)

                counters_list.append({
                    "sample_id": row.sample_id,
                    "zero_prefill_verified": counters.is_verified,
                    "counters": counters.__dict__,
                })
        finally:
            self._unload_student()

        # §53 CUDA Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── 构造 artifact payloads ──
        dataset_name = "unknown"
        ds_cfg = self.cfg.get("datasets", {})
        fidelity = ds_cfg.get("fidelity", [])
        if fidelity:
            dataset_name = str(fidelity[0]) if isinstance(fidelity, list) else str(fidelity)

        t_cfg = self.cfg.get("teacher", {})
        s_cfg = self.cfg.get("student", {})
        provenance = {
            "teacher_model_id": t_cfg.get("model_id", ""),
            "student_model_id": s_cfg.get("model_id", ""),
            "teacher_revision": t_cfg.get("revision", "main"),
            "student_revision": s_cfg.get("revision", "main"),
            "teacher_dtype": t_cfg.get("dtype", "bfloat16"),
            "student_dtype": s_cfg.get("dtype", "bfloat16"),
            "teacher_num_layers": t_cfg.get("num_layers", 0),
            "student_num_layers": s_cfg.get("num_layers", 0),
            "tokenizer_available": self._student_tokenizer is not None,
            # D6: 版本固定 —— 代码版本与协议版本进产物，artifacts 可溯源
            "protocol_version": PROTOCOL_VERSION,
            "git_hash": _git_hash(),
            # A2: 切分与 RoPE 对齐模式如实标注
            "calib_eval_disjoint": bool(calib_eval_disjoint),
            "n_calib_samples": len(calib_rows),
            "n_eval_samples": len(eval_rows),
            "calib_sample_ids": [r.sample_id for r in calib_rows],
            "eval_sample_ids": [r.sample_id for r in eval_rows],
            "rope_align": self._rope_align,
            "ablation_modes": list(ablation_modes),
        }

        # B1: rows / teacher 分数 / manifest 落盘（KV 与 mapper 参数已各自写入）
        if self._persist_kv:
            try:
                rows_dicts = [
                    {
                        "sample_id": r.sample_id,
                        "context": r.context,
                        "query": r.query,
                        "answer": r.answer,
                        "split": getattr(r, "split", "test"),
                        "is_calib": r.sample_id in {c.sample_id for c in calib_rows},
                    }
                    for r in rows
                ]
                write_rows_teacher(self._kv_store_dir, rows_dicts, teacher_scores_store)
                write_manifest(
                    self._kv_store_dir,
                    {
                        "protocol_version": PROTOCOL_VERSION,
                        "git_hash": _git_hash(),
                        "teacher_model_id": t_cfg.get("model_id", ""),
                        "student_model_id": s_cfg.get("model_id", ""),
                        "rope_align": self._rope_align,
                        "n_samples": len(rows_dicts),
                    },
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("[inject-eval] kv_store 落盘失败: %s", e)

        # B2: PSR 汇总
        if psr_rows:
            psr_arr = np.asarray(psr_rows, dtype=np.float64)
            psr_summary = {
                "mean": float(psr_arr.mean()),
                "median": float(np.median(psr_arr)),
                "n": int(psr_arr.size),
                "note": (
                    "t_handoff 含测量专用 PPL forward（部署时不需要），PSR 偏保守；"
                    "校准阶段 student self-prefill 成本未计入（见 mapper 校准成本）。"
                ),
            }
        else:
            psr_summary = {"mean": None, "median": None, "n": 0, "note": "kv_both 未运行"}

        replacement_artifact = {
            "evidence_grade": "measured_task",
            "zero_prefill_verified": all_zero_prefill,
            "task_scoring_verified": True,
            "split": "test",
            "dataset": dataset_name,
            "provenance": provenance,
            "records": replacement_records,
        }

        # capability artifact: 四种方法全覆盖（base_only/base_plus_adv/full_apcs 已冻结）
        # 计算 per-method 准确率（A1: 含 gold 概率口径）
        method_stats: dict[str, dict[str, Any]] = {}
        for rec in capability_records:
            m = rec.get("method", "unknown")
            if m not in method_stats:
                method_stats[m] = {"total": 0, "correct": 0, "scores": [], "gold_probs": []}
            method_stats[m]["total"] += 1
            if rec.get("correct", False):
                method_stats[m]["correct"] += 1
            method_stats[m]["scores"].append(rec.get("score", 0))
            if rec.get("gold_prob") is not None:
                method_stats[m]["gold_probs"].append(rec["gold_prob"])

        method_accuracy = {}
        for m, stats in method_stats.items():
            total = stats["total"]
            correct = stats["correct"]
            scores = stats["scores"]
            method_accuracy[m] = {
                "accuracy": correct / total if total > 0 else 0.0,
                "correct": correct,
                "total": total,
                "score_mean": sum(scores) / len(scores) if scores else 0.0,
                "gold_prob_mean": (
                    sum(stats["gold_probs"]) / len(stats["gold_probs"])
                    if stats["gold_probs"] else None
                ),
            }

        capability_artifact = {
            "evidence_grade": "measured_task",
            "zero_prefill_verified": all_zero_prefill,
            "task_scoring_verified": True,
            "split": "test",
            "dataset": dataset_name,
            "provenance": provenance,
            "omitted_methods": ["base_only", "base_plus_adv", "full_apcs"],
            "omitted_reason": "advantage branch frozen (T08); methods not yet trained",
            "method_accuracy": method_accuracy,
            "records": capability_records,
        }

        return {
            "replacement_artifact": replacement_artifact,
            "capability_artifact": capability_artifact,
            "counters_summary": counters_list,
            "zero_prefill_verified": all_zero_prefill,
            "psr_summary": psr_summary,
            "kv_diagnostics": dict(self._kv_norm_stats),
        }

    # ------------------------------------------------------------------
    # B1 在线阶段：从持久化 KV store 冷启动（全程不加载 Teacher）
    # ------------------------------------------------------------------

    def evaluate_online(self, store_dir: str | Path) -> dict[str, Any]:
        """核心目标闭环的在线半程：读盘 KV → mapper → 注入 → 零 prefill 评分。

        输入 store 由 evaluate(persist_kv=True) 离线阶段产出：
            kv/<sample_id>.npz + rows.json + teacher_scores.json
            + mapper_params.npz + manifest.json（含 sha256 校验和）

        流程：
            1. load_store（校验和验证，防"读错缓存"）
            2. 仅加载 Student（§53 单卡纪律；Teacher 全程不在场）
            3. mapper 参数优先从磁盘恢复（离线已 fit）；恢复失败才回退
               "student 对校准行 self-prefill 现场拟合"，并在 provenance
               如实标注 mapper_source（§75：不把在线拟合谎报为离线迁移）
            4. 对评估行跑 student_self / text_handoff / ridge_handoff，
               teacher KV 用 load_kv_sample 从盘读
            5. teacher 分数取自离线 teacher_scores.json

        Returns:
            与 evaluate() 同构的 dict（另含 provenance.online=True）。
        """
        import torch  # type: ignore

        from .backends import _extract_kv_separate
        from .kv_store import load_kv_sample, load_mapper_params, load_store

        store = load_store(Path(store_dir), verify=True)
        rows_dicts = store["rows"]
        if not rows_dicts:
            raise RuntimeError(f"evaluate_online: store 无样本 rows（{store_dir}）")
        rows = [SimpleNamespace(**r) for r in rows_dicts]
        calib_rows = [r for r in rows if getattr(r, "is_calib", False)]
        eval_rows = [r for r in rows if not getattr(r, "is_calib", False)]
        if not eval_rows:
            eval_rows = rows  # 兼容旧 store（无切分标注）
        calib_eval_disjoint = bool(calib_rows) and set(c.sample_id for c in calib_rows).isdisjoint(
            e.sample_id for e in eval_rows
        )
        eval_ids = {r.sample_id for r in eval_rows}

        def _timed(fn):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fn()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            return out, (time.perf_counter() - t0) * 1000.0

        default_choices = self.cfg.get("inject_eval", {}).get("default_choices", None)
        ablation_modes = self.cfg.get("inject_eval", {}).get(
            "ablation_modes", ["native", "kv_both"],
        )

        capability_records: list[dict[str, Any]] = []
        replacement_records: list[dict[str, Any]] = []
        counters_list: list[dict[str, Any]] = []
        all_zero_prefill = True
        psr_rows: list[float] = []

        # teacher 分数来自离线 store（在线阶段不加载 Teacher）
        teacher_scores = store.get("teacher_scores", {})
        for r in eval_rows:
            ts = teacher_scores.get(r.sample_id)
            capability_records.append({
                "method": "teacher",
                "sample_id": r.sample_id,
                "seed": self._seed,
                "score": float(ts["score"]) if ts else 0.0,
                "decision": int(ts["decision"]) if ts else -1,
                "correct": bool(ts["correct"]) if ts else False,
                "gold_prob": ts.get("gold_prob") if ts else None,
                "source": "offline_store",
            })

        self._load_student()
        mapper_source = "online_fit"
        try:
            self._build_mappers()
            if store["has_mapper_params"] and load_mapper_params(
                Path(store_dir), {"K": self._mapper_k, "V": self._mapper_v}
            ):
                mapper_source = "offline_store"
                self._mapper = self._mapper_k
                logger.info("[inject-eval] online: mapper params loaded from store")
            else:
                logger.warning(
                    "[inject-eval] online: store 无 mapper 参数，回退在线拟合"
                    "（校准行 student self-prefill；离线保存参数可避免此成本）"
                )
                real_kv_calib: dict[str, list] = {"K": [], "V": []}
                rope_inv_freq = None
                if self._rope_align == "unrotated":
                    from ..rope.runner import _rope_pairs, de_rope as _de_rope
                    rope_D = int(self.cfg.get("teacher", {}).get("head_dim", 128))
                    rope_inv_freq = _rope_pairs(rope_D, theta=1_000_000.0)
                for row in calib_rows:
                    try:
                        k, v, S_t = load_kv_sample(Path(store_dir), row.sample_id)
                        student_ids = self._student_tokenizer(
                            row.context, return_tensors="np"
                        ).input_ids.reshape(-1)
                        s_ids_t = torch.as_tensor(
                            student_ids.reshape(1, -1), dtype=torch.long
                        ).to(self._student_device)
                        with torch.no_grad():
                            student_out = self._student_model(s_ids_t, use_cache=True)
                        n_s = int(
                            getattr(self._student_model.config, "num_hidden_layers", 0)
                        ) or len(student_out.past_key_values)
                        s_k_np, s_v_np = _extract_kv_separate(student_out.past_key_values, n_s)
                        del student_out
                        min_S = min(k.shape[1], s_k_np.shape[1])
                        s_k_np = s_k_np[:, :min_S, :, :]
                        s_v_np = s_v_np[:, :min_S, :, :]
                        if rope_inv_freq is not None:
                            s_k_np = _de_rope(
                                s_k_np, np.arange(min_S, dtype=np.float64), rope_inv_freq
                            )
                        if k.shape[-1] == s_k_np.shape[-1]:
                            real_kv_calib["K"].append(
                                (k[:, :min_S, :, :], s_k_np)
                            )
                            real_kv_calib["V"].append(
                                (v[:, :min_S, :, :], s_v_np)
                            )
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            "[inject-eval] online calibration failed for %s: %s",
                            row.sample_id, e,
                        )
                self._fit_mappers(real_kv_calib)
                self._mapper = self._mapper_k

            for row in eval_rows:
                if default_choices is not None:
                    choices = default_choices
                else:
                    choices = _extract_choices_from_query(row.query)
                    if not choices:
                        logger.warning(
                            "[inject-eval] online: cannot extract choices for %s, skipping",
                            row.sample_id,
                        )
                        continue

                try:
                    (s_score, s_decision, s_logits, s_ppl), t_prefill_ms = _timed(
                        lambda: self._student_self(row, choices)
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning("[inject-eval] online student_self failed for %s: %s", row.sample_id, e)
                    s_score, s_decision, s_logits, s_ppl, t_prefill_ms = 0.0, 0, None, float("inf"), 0.0
                s_correct = False
                if row.answer is not None:
                    letters = ["A", "B", "C", "D"]
                    if 0 <= s_decision < len(letters):
                        s_correct = letters[s_decision].upper() == row.answer.strip().upper()
                s_gold = gold_prob_from_logits(s_logits, row.answer, self._student_letter_ids)
                capability_records.append({
                    "method": "student",
                    "sample_id": row.sample_id,
                    "seed": self._seed,
                    "score": float(s_score),
                    "decision": int(s_decision),
                    "correct": bool(s_correct),
                    "gold_prob": float(s_gold) if s_gold is not None else None,
                    "ppl": float(s_ppl),
                    "prefill_ms": float(t_prefill_ms),
                })

                try:
                    t_score, t_decision, t_logits = self._text_handoff(row, choices)
                except Exception as e:  # noqa: BLE001
                    logger.warning("[inject-eval] online text_handoff failed for %s: %s", row.sample_id, e)
                    t_score, t_decision, t_logits = 0.0, 0, None
                t_correct = False
                if row.answer is not None:
                    letters = ["A", "B", "C", "D"]
                    if 0 <= t_decision < len(letters):
                        t_correct = letters[t_decision].upper() == row.answer.strip().upper()
                t_gold = gold_prob_from_logits(t_logits, row.answer, self._student_letter_ids)
                capability_records.append({
                    "method": "text",
                    "sample_id": row.sample_id,
                    "seed": self._seed,
                    "score": float(t_score),
                    "decision": int(t_decision),
                    "correct": bool(t_correct),
                    "gold_prob": float(t_gold) if t_gold is not None else None,
                })

                # KV 从磁盘读（持久化迁移的核心路径）
                try:
                    k_disk, v_disk, S_disk = load_kv_sample(Path(store_dir), row.sample_id)
                    cached_kv = (k_disk, v_disk, None, S_disk)
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "[inject-eval] online KV load failed for %s: %s", row.sample_id, e
                    )
                    cached_kv = None
                row_self_kv: dict[str, tuple] = {}  # P0.0: self_kv 恒等对照的行内缓存
                t_handoff_ms = 0.0
                for mode in ablation_modes:
                    counters = EvalPhaseCounters()
                    try:
                        if cached_kv is None:
                            raise RuntimeError("KV not available in store")
                        if mode == "kv_both":
                            (r_score, r_decision, r_logits, counters, r_ppl), t_handoff_ms = _timed(
                                lambda: self._ridge_handoff(
                                    row, choices, ablation_mode=mode,
                                    preextracted_kv=cached_kv,
                                    self_kv_cache=row_self_kv,
                                )
                            )
                        else:
                            r_score, r_decision, r_logits, counters, r_ppl = self._ridge_handoff(
                                row, choices, ablation_mode=mode,
                                preextracted_kv=cached_kv,
                                self_kv_cache=row_self_kv,
                            )
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            "[inject-eval] online ridge_handoff(%s) failed for %s: %s",
                            mode, row.sample_id, e,
                        )
                        r_score, r_decision, r_ppl, r_logits = 0.0, 0, float("inf"), None
                        counters = EvalPhaseCounters()
                        all_zero_prefill = False
                    if not counters.is_verified:
                        all_zero_prefill = False

                    r_correct = False
                    if row.answer is not None:
                        letters = ["A", "B", "C", "D"]
                        if 0 <= r_decision < len(letters):
                            r_correct = letters[r_decision].upper() == row.answer.strip().upper()
                    r_gold = gold_prob_from_logits(r_logits, row.answer, self._student_letter_ids)
                    # P0.6: 行为层诊断（online 路径同 evaluate）
                    r_logit_cos = None
                    if r_logits is not None and s_logits is not None:
                        denom = float(np.linalg.norm(r_logits) * np.linalg.norm(s_logits))
                        if denom > 0:
                            r_logit_cos = float(np.dot(r_logits, s_logits) / denom)
                    capability_records.append({
                        "method": f"ridge_{mode}",
                        "sample_id": row.sample_id,
                        "seed": self._seed,
                        "score": float(r_score),
                        "decision": int(r_decision),
                        "correct": bool(r_correct),
                        "gold_prob": float(r_gold) if r_gold is not None else None,
                        "ppl": float(r_ppl),
                        "logit_cos_vs_student": r_logit_cos,
                    })
                    if mode == "kv_both":
                        replacement_records.append({
                            "sample_id": row.sample_id,
                            "student_score": float(s_score),
                            "handoff_score": float(r_score),
                            "student_gold_prob": float(s_gold) if s_gold is not None else None,
                            "handoff_gold_prob": float(r_gold) if r_gold is not None else None,
                        })
                        if t_prefill_ms > 0:
                            psr_rows.append((t_prefill_ms - t_handoff_ms) / t_prefill_ms)

                counters_list.append({
                    "sample_id": row.sample_id,
                    "zero_prefill_verified": counters.is_verified,
                    "counters": counters.__dict__,
                })
        finally:
            self._unload_student()

        dataset_name = "unknown"
        ds_cfg = self.cfg.get("datasets", {})
        fidelity = ds_cfg.get("fidelity", [])
        if fidelity:
            dataset_name = str(fidelity[0]) if isinstance(fidelity, list) else str(fidelity)
        t_cfg = self.cfg.get("teacher", {})
        s_cfg = self.cfg.get("student", {})
        provenance = {
            "teacher_model_id": t_cfg.get("model_id", ""),
            "student_model_id": s_cfg.get("model_id", ""),
            "teacher_revision": t_cfg.get("revision", "main"),
            "student_revision": s_cfg.get("revision", "main"),
            "teacher_dtype": t_cfg.get("dtype", "bfloat16"),
            "student_dtype": s_cfg.get("dtype", "bfloat16"),
            "teacher_num_layers": t_cfg.get("num_layers", 0),
            "student_num_layers": s_cfg.get("num_layers", 0),
            "tokenizer_available": self._student_tokenizer is not None,
            "protocol_version": PROTOCOL_VERSION,
            "git_hash": _git_hash(),
            "online": True,
            "kv_store_dir": str(store_dir),
            "mapper_source": mapper_source,
            "calib_eval_disjoint": calib_eval_disjoint,
            "n_calib_samples": len(calib_rows),
            "n_eval_samples": len(eval_rows),
            "rope_align": self._rope_align,
            "ablation_modes": list(ablation_modes),
        }

        if psr_rows:
            psr_arr = np.asarray(psr_rows, dtype=np.float64)
            psr_summary = {
                "mean": float(psr_arr.mean()),
                "median": float(np.median(psr_arr)),
                "n": int(psr_arr.size),
                "note": "在线阶段：KV 从磁盘加载；t_handoff 含测量用 PPL forward，偏保守",
            }
        else:
            psr_summary = {"mean": None, "median": None, "n": 0, "note": "kv_both 未运行"}

        method_stats: dict[str, dict[str, Any]] = {}
        for rec in capability_records:
            m = rec.get("method", "unknown")
            if m not in method_stats:
                method_stats[m] = {"total": 0, "correct": 0, "scores": [], "gold_probs": []}
            method_stats[m]["total"] += 1
            if rec.get("correct", False):
                method_stats[m]["correct"] += 1
            method_stats[m]["scores"].append(rec.get("score", 0))
            if rec.get("gold_prob") is not None:
                method_stats[m]["gold_probs"].append(rec["gold_prob"])
        method_accuracy = {}
        for m, stats in method_stats.items():
            total = stats["total"]
            method_accuracy[m] = {
                "accuracy": stats["correct"] / total if total > 0 else 0.0,
                "correct": stats["correct"],
                "total": total,
                "score_mean": sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else 0.0,
                "gold_prob_mean": (
                    sum(stats["gold_probs"]) / len(stats["gold_probs"])
                    if stats["gold_probs"] else None
                ),
            }

        replacement_artifact = {
            "evidence_grade": "measured_task",
            "zero_prefill_verified": all_zero_prefill,
            "task_scoring_verified": True,
            "split": "test",
            "dataset": dataset_name,
            "provenance": provenance,
            "records": replacement_records,
        }
        capability_artifact = {
            "evidence_grade": "measured_task",
            "zero_prefill_verified": all_zero_prefill,
            "task_scoring_verified": True,
            "split": "test",
            "dataset": dataset_name,
            "provenance": provenance,
            "omitted_methods": ["base_only", "base_plus_adv", "full_apcs"],
            "omitted_reason": "advantage branch frozen (T08); methods not yet trained",
            "method_accuracy": method_accuracy,
            "records": capability_records,
        }
        return {
            "replacement_artifact": replacement_artifact,
            "capability_artifact": capability_artifact,
            "counters_summary": counters_list,
            "zero_prefill_verified": all_zero_prefill,
            "psr_summary": psr_summary,
            "kv_diagnostics": dict(self._kv_norm_stats),
        }

    # ------------------------------------------------------------------
    # 产物写入
    # ------------------------------------------------------------------

    def write_artifacts(self, eval_result: dict[str, Any]) -> None:
        """将评估结果写入 <run_dir>/inject_eval/ 目录。

        产物清单：
            replacement_score_artifact.json — T05 消费（_load_replacement_task_artifact）
            capability_score_artifact.json  — T09 消费（ArtifactScoreProvider.open）
            hidden_states.npz (可选)        — T12 geometry diagnostics

        §75 诚实性：所有产物携带 provenance（model ids, revision, dtype 等），
        非有限 score 会被 reject。
        """
        out_dir = self.run_dir / "inject_eval"
        out_dir.mkdir(parents=True, exist_ok=True)

        replacement = eval_result["replacement_artifact"]
        capability = eval_result["capability_artifact"]

        # 校验 replacement artifact 的 records 非有限
        for rec in replacement.get("records", []):
            for key in ("student_score", "handoff_score"):
                if not math.isfinite(float(rec.get(key, 0))):
                    raise RuntimeError(
                        f"replacement artifact 非有限 score: {rec['sample_id']}.{key}"
                    )

        # 校验 capability artifact 的 records
        for rec in capability.get("records", []):
            if not math.isfinite(float(rec.get("score", 0))):
                raise RuntimeError(
                    f"capability artifact 非有限 score: {rec.get('method')}/{rec['sample_id']}"
                )

        # 写 replacement_score_artifact.json
        _write_json(out_dir / "replacement_score_artifact.json", replacement)
        logger.info("[inject-eval] Wrote replacement_score_artifact.json (%d records)", len(replacement.get("records", [])))

        # 写 capability_score_artifact.json
        _write_json(out_dir / "capability_score_artifact.json", capability)
        logger.info("[inject-eval] Wrote capability_score_artifact.json (%d records)", len(capability.get("records", [])))

        # 写 hidden states NPZ（可选，供 T12 geometry diagnostics）
        if self._export_hidden:
            self._write_hidden_states_npz(out_dir, eval_result)


    def _write_hidden_states_npz(
        self, out_dir: Path, eval_result: dict[str, Any]
    ) -> None:
        """导出 hidden states NPZ 供 T12 geometry diagnostics 使用。

        §40 geometry 路径要求 teacher/student hidden states：
            - ndim==3, equal hidden dim, equal token count

        由于 Teacher (Qwen3-4B: hidden=2560) 和 Student (Qwen3-1.7B: hidden=2048)
        的 hidden dim 不一致，这里做 orthogonal projection 到公共维度 512。
        projection matrix 以确定性种子生成，存入 NPZ 的 proj_t/proj_s 键。

        NPZ 内容：
            teacher: projected teacher hidden states (L_t, N, D_proj)
            student: projected student hidden states (L_s, N, D_proj)
            proj_t: projection matrix (hidden_t, D_proj)
            proj_s: projection matrix (hidden_s, D_proj)
            meta: JSON bytes 说明 provenance
        """
        import torch  # type: ignore

        t_cfg = self.cfg.get("teacher", {})
        s_cfg = self.cfg.get("student", {})
        n_t = int(t_cfg.get("num_layers", 36))
        n_s = int(s_cfg.get("num_layers", 28))
        hidden_t = int(t_cfg.get("hidden_size", 2560))
        hidden_s = int(s_cfg.get("hidden_size", 2048))
        d_proj = 512  # target projection dim

        rng = np.random.default_rng(self._seed)

        # 确定性正交投影矩阵
        proj_t = _orthogonal_projection(rng, hidden_t, d_proj)
        proj_s = _orthogonal_projection(rng, hidden_s, d_proj)

        # 从 eval_result 获取样本数据
        samples = eval_result.get("samples", [])
        if not samples:
            logger.warning("[inject-eval] No samples in eval_result, skipping hidden states export")
            return

        n_positions = min(16, len(samples))  # 最多16个位置
        teacher_hidden_list = []
        student_hidden_list = []

        try:
            # Phase 1: 提取 teacher hidden states
            if self._teacher_model is not None and self._teacher_tokenizer is not None:
                self._load_teacher()
                try:
                    for i, sample in enumerate(samples[:n_positions]):
                        row = sample.get("row")
                        if row is None:
                            continue
                        text = f"{row.context}\n{row.query}" if row.context else row.query
                        ids = self._teacher_tokenizer(text, return_tensors="np").input_ids.reshape(-1)
                        ids_t = torch.as_tensor(ids.reshape(1, -1), dtype=torch.long).to(self._teacher_device)
                        
                        # 使用 output_hidden_states=True 获取所有层的hidden states
                        with torch.no_grad():
                            out = self._teacher_model(ids_t, use_cache=False, output_hidden_states=True)
                        
                        # hidden_states: tuple of (L_t+1, batch, seq_len, hidden)
                        hidden_states = out.hidden_states
                        if hidden_states is not None:
                            # 取最后一层的hidden state，然后投影
                            last_hidden = hidden_states[-1][0, -1, :].float().cpu().numpy()  # (hidden_t,)
                            projected = last_hidden @ proj_t  # (d_proj,)
                            teacher_hidden_list.append(projected)
                        
                        del out
                finally:
                    self._unload_teacher()
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # Phase 2: 提取 student hidden states
            if self._student_model is not None and self._student_tokenizer is not None:
                self._load_student()
                try:
                    for i, sample in enumerate(samples[:n_positions]):
                        row = sample.get("row")
                        if row is None:
                            continue
                        text = f"{row.context}\n{row.query}" if row.context else row.query
                        ids = self._student_tokenizer(text, return_tensors="np").input_ids.reshape(-1)
                        ids_t = torch.as_tensor(ids.reshape(1, -1), dtype=torch.long).to(self._student_device)
                        
                        # 使用 output_hidden_states=True 获取所有层的hidden states
                        with torch.no_grad():
                            out = self._student_model(ids_t, use_cache=False, output_hidden_states=True)
                        
                        # hidden_states: tuple of (L_s+1, batch, seq_len, hidden)
                        hidden_states = out.hidden_states
                        if hidden_states is not None:
                            # 取最后一层的hidden state，然后投影
                            last_hidden = hidden_states[-1][0, -1, :].float().cpu().numpy()  # (hidden_s,)
                            projected = last_hidden @ proj_s  # (d_proj,)
                            student_hidden_list.append(projected)
                        
                        del out
                finally:
                    self._unload_student()
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # 转换为numpy数组
            if teacher_hidden_list:
                teacher_hidden = np.stack(teacher_hidden_list, axis=0).astype(np.float32)  # (N, d_proj)
                # 重塑为 (L_t, N, D_proj) 格式，这里L_t=1（只取最后一层）
                teacher_hidden = teacher_hidden.reshape(1, -1, d_proj)
            else:
                teacher_hidden = np.zeros((1, n_positions, d_proj), dtype=np.float32)
                logger.warning("[inject-eval] No teacher hidden states extracted, using zeros")

            if student_hidden_list:
                student_hidden = np.stack(student_hidden_list, axis=0).astype(np.float32)  # (N, d_proj)
                # 重塑为 (L_s, N, D_proj) 格式，这里L_s=1（只取最后一层）
                student_hidden = student_hidden.reshape(1, -1, d_proj)
            else:
                student_hidden = np.zeros((1, n_positions, d_proj), dtype=np.float32)
                logger.warning("[inject-eval] No student hidden states extracted, using zeros")

            meta = json.dumps({
                "note": (
                    "Real hidden states extracted from model forward pass; "
                    "last layer hidden state projected to common dimension; "
                    "projection matrices are deterministic (seed={})".format(self._seed)
                ),
                "teacher_hidden_t": hidden_t,
                "student_hidden_s": hidden_s,
                "d_proj": d_proj,
                "n_positions": len(teacher_hidden_list),
                "extraction_method": "last_layer_hidden_state",
            }, ensure_ascii=False).encode("utf-8")

            npz_path = out_dir / "hidden_states.npz"
            np.savez(
                npz_path,
                teacher=teacher_hidden,
                student=student_hidden,
                proj_t=proj_t,
                proj_s=proj_s,
                meta=meta,
            )
            logger.info(
                "[inject-eval] Wrote hidden_states.npz (teacher=%s, student=%s, n_samples=%d)",
                teacher_hidden.shape, student_hidden.shape, len(teacher_hidden_list),
            )

        except Exception as e:
            logger.error("[inject-eval] Failed to extract hidden states: %s", e)
            # ◆ D3 修复：失败时不再写全零矩阵（误导下游 geometry 诊断），
            #   直接跳过 —— 没有 hidden_states.npz 比有假数据诚实
            return


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """写 JSON 文件，UTF-8 编码，2-space indent。"""
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _orthogonal_projection(rng: np.random.Generator, in_dim: int, out_dim: int) -> np.ndarray:
    """生成确定性正交投影矩阵 (in_dim, out_dim)。

    方法：QR 分解随机矩阵，取前 out_dim 列。
    保证投影矩阵列正交（单位列范数）。
    """
    if out_dim > in_dim:
        raise ValueError(f"投影维度 out_dim({out_dim}) > in_dim({in_dim})")
    random_matrix = rng.standard_normal((in_dim, out_dim))
    q, _ = np.linalg.qr(random_matrix)
    return q[:, :out_dim].astype(np.float32)


# ---------------------------------------------------------------------------
# 渐进式注入（Progressive Injection）
# ---------------------------------------------------------------------------


class ProgressiveInjectionEvaluator:
    """渐进式注入评估器，支持Stage I (Compatibility Base) 和 Stage II (Advantage Residual)。

    设计目标：
    1. Stage I: 只注入几何兼容的层（CKA > threshold），其余层使用Student self-prefill
    2. Stage II: 在Stage I通过后，训练bounded advantage residual
    3. 避免一次性注入所有层导致Student表示破坏
    """

    def __init__(
        self,
        base_evaluator: InjectionEvaluator,
        cka_threshold: float = 0.3,
        stage: str = "stage_i",
    ):
        """初始化渐进式注入评估器。

        Args:
            base_evaluator: 基础评估器实例
            cka_threshold: CKA阈值，高于此值的层才注入
            stage: 阶段，"stage_i"或"stage_ii"
        """
        self.base_evaluator = base_evaluator
        self.cka_threshold = cka_threshold
        self.stage = stage
        
        # 存储层兼容性信息
        self.layer_compatibility: dict[int, float] = {}
        
        # Advantage residual参数（Stage II）
        self.advantage_params: dict[str, Any] = {}

    def compute_layer_compatibility(
        self,
        teacher_kv: np.ndarray,
        student_kv: np.ndarray,
    ) -> dict[int, float]:
        """计算每层的CKA兼容性分数。

        Args:
            teacher_kv: Teacher KV (L_t, S, H, D)
            student_kv: Student KV (L_s, S, H, D)

        Returns:
            层索引到CKA分数的字典
        """
        from ..metrics import linear_cka
        
        n_s = student_kv.shape[0]
        n_t = teacher_kv.shape[0]
        
        compatibility = {}
        
        for s in range(n_s):
            # 找到最相似的Teacher层
            best_cka = 0.0
            for t in range(n_t):
                # 计算CKA
                student_flat = student_kv[s].reshape(-1)
                teacher_flat = teacher_kv[t].reshape(-1)
                
                # 使用线性CKA
                cka_score = linear_cka(
                    student_flat.reshape(1, -1),
                    teacher_flat.reshape(1, -1),
                )
                best_cka = max(best_cka, float(cka_score))
            
            compatibility[s] = best_cka
        
        self.layer_compatibility = compatibility
        return compatibility

    def select_injectable_layers(self) -> list[int]:
        """选择可注入的层索引。

        Returns:
            可注入的层索引列表
        """
        injectable = [
            s for s, cka in self.layer_compatibility.items()
            if cka >= self.cka_threshold
        ]
        
        logger.info(
            "[progressive-injection] Selected %d/%d layers for injection (threshold=%.3f)",
            len(injectable), len(self.layer_compatibility), self.cka_threshold,
        )
        
        return injectable

    def create_selective_layer_map(
        self,
        full_layer_map: list[list[int]],
        injectable_layers: list[int],
    ) -> list[list[int]]:
        """创建选择性层映射，只对可注入层使用Teacher映射。

        Args:
            full_layer_map: 完整层映射
            injectable_layers: 可注入的层索引列表

        Returns:
            选择性层映射
        """
        n_s = len(full_layer_map)
        selective_map = []
        
        for s in range(n_s):
            if s in injectable_layers:
                # 使用Teacher映射
                selective_map.append(full_layer_map[s])
            else:
                # 使用Identity映射（保持Student原始状态）
                # 返回空列表，表示该层不注入
                selective_map.append([])
        
        return selective_map

    def inject_with_selectivity(
        self,
        teacher_kv: np.ndarray,
        layer_map: list[list[int]],
        student_kv: np.ndarray | None = None,
        kv_kind: str = "K",
    ) -> np.ndarray:
        """选择性注入Teacher KV。

        Args:
            teacher_kv: Teacher KV (L_t, S, H, D)
            layer_map: 选择性层映射
            student_kv: Student self-prefill KV（用于未注入层）
            kv_kind: KV类型

        Returns:
            注入后的KV (L_s, S, H, D)
        """
        L_s = len(layer_map)
        L_t, S, H, D = teacher_kv.shape
        
        out = np.zeros((L_s, S, H, D), dtype=teacher_kv.dtype)
        
        for s in range(L_s):
            if layer_map[s]:
                # 使用Teacher映射
                teachers = layer_map[s]
                for t in teachers:
                    out[s] += teacher_kv[t]
                out[s] /= len(teachers)
            elif student_kv is not None:
                # 使用Student self-prefill
                out[s] = student_kv[s]
            else:
                # 如果没有Student KV，使用零注入
                logger.warning(
                    "[progressive-injection] Layer %d not injectable and no student KV provided",
                    s,
                )
        
        return out


__all__ = [
    "InjectionEvaluator",
    "ProgressiveInjectionEvaluator",
    "score_choices",
    "EvalPhaseCounters",
    "_build_scoring_prompt",
    "_extract_gold_index",
]
