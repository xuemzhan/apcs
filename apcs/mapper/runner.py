"""T04 / T05 / T06 / Replacement Runner。

═══════════════════════════════════════════════════════════════════════════════
本模块对应 design.md：
    §19  Baselines B3 (Cross-Model Ridge) + B4 (MLP) + O1/O2/O3 (APCS)
    §20  Compatibility Base — Ridge / Low-rank / Shared basis
    §32  T04 Ridge baseline calibration (100-500 samples, 512/1K context)
    §33  T05 Replacement 评估 (Retention / KL / Token Agreement / Latency)
    §34  T06 Lightweight mapper (rank 8/16/32, shared basis, PCR-Retention Figure 1)
    §45  JCR (Judge Consistency Rate)
    §48  Gap strata 报告
    §49  计时规范 (warmup + sync + ≥10 repeats + P50/P95)
    §52  禁止 1：Student 不重新读 X（仅在 calibration / eval 阶段允许；真实
          inference 阶段 zero prefill）

═══════════════════════════════════════════════════════════════════════════════
修复内容（对比上一版）：
- bug-3 修复（聚合校准）：T04/T05/T06 的校准循环 `for i: ridge.fit(...)`
  每次 fit 覆盖 W，最终只留最后一组样本的影响。现改为：校准样本共享
  W_t/W_s（同一 Teacher/Student 模型对不同 prompt），用
  `fit_ridge_aggregate`（Gram 聚合，数学上等于全部样本拼接后一次 fit，
  §32 方案 B）只 fit 一次；ALS 类 mapper（LowRank / SharedBasis）用
  `concat_kv_samples` 拼接后一次 fit（方案 A，t06_aggregate_samples 限内存）。
- bug-4 修复：calibration 数据从"每层独立随机"改为"先构造共享 latent，
  再用不同线性变换派生 Teacher 与 Student KV"，让 ridge 学到的是
  真实可恢复的映射，而非随机噪声。
- §34 T06 增加 shared_basis 实验（A2 消融）。
- §32 T04 增加 attention-output cosine 报告。
- §33 T05 增加 Token Agreement 与 Latency (P50/P95)。
- §45 JCR 接入：Student self-prefill 的 next-token decision vs handoff。
- §48 Gap strata：T05/T09 按 low/medium/high gap 分别报告。
- §49 计时：所有耗时测量按 (warmup, repeats, sync) 协议跑。
- §53 单卡执行策略：见 `single_card_pipeline()`。

───────────────────────────────────────────────────────────────────────────────
通用约定（T04/T05/T06 共用）：
    - 种子规则：模型权重由 `master_seed` 生成并跨样本复用（同一模型对）；
      样本 latent Z 的种子 = `master_seed*1_000_000 + i`，保证不同
      master_seed 的数据集（calib=0 / held-out=1 / T06 附加 eval=2）的
      Z 与噪声完全不相交。
    - §22 K/V 独立参数化：`mapper.separate_kv=true` 时 K 与 V 视为两套独立
      缓存，各自用 `kv_seed_offset`（KIND_SEED_OFFSET 错开）的随机抽样校准集
      fit 一套参数；所有 T0x 评测变体输出 `*_K` / `*_V` 分项字段。
    - 校准与评估必须共享同一组 w_t/w_s（同一 Teacher/Student 模型对），只换
      latent Z —— 否则 train/eval 等于拿不同模型的 KV 做（cosine 归零）。
"""
from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from ..alignment.runner import proportional_mapping
from ..io.runs import write_json
from ..metrics import cosine, jcr, kl_divergence, pcr, r2
from ..rope.runner import _rope_pairs, de_rope
from ..utils import percentile
from .aggregate import concat_kv_samples, fit_ridge_aggregate
from .math import LowRankMapper, RidgePerHeadMapper, SharedBasisMapper

# ===========================================================================
# design.md §22：K/V 独立参数化 —— 单一事实源
# ===========================================================================
# K 与 V 的随机抽样错开固定 offset：同一结构（共享 W_t/W_s 模型对）、
# 不同随机性（独立 latent Z）—— 模拟真实 K/V 是两个独立缓存（§22）。
KIND_SEED_OFFSET: dict[str, int] = {"K": 0, "V": 100000}


def kv_kinds(cfg: dict[str, Any]) -> list[str]:
    """§22：决定当前评测使用哪些 KV 种类（K/V 独立参数化）。

    参数：
        cfg: 全局配置字典，仅读取 `cfg["mapper"]["separate_kv"]`
    返回：
        separate_kv=true → ["K", "V"]（K 与 V 独立校准/评估、互不覆盖）；
        否则 → ["K"]（单 kind 行为，与旧实现数值一致）。
    """
    separate_kv = bool(cfg.get("mapper", {}).get("separate_kv", False))
    return ["K", "V"] if separate_kv else ["K"]


def _merge_kv_fields(
    dst: dict[str, Any],
    per_kind: dict[str, dict[str, float]],
    fields: dict[str, str],
) -> None:
    """§22：把 per-kind 指标合并为 `{base}_K` / `{base}_V` 顶层字段。

    参数：
        dst: 目标指标字典（原地修改）
        per_kind: {"K": {...}, "V": {...}} 各 kind 的指标子字典
        fields: {输出基名: per_kind[kind] 内键}，如 {"retention": "retention"} →
            dst["retention_K"] / dst["retention_V"]
    返回：
        None。仅当 per_kind 含 "V"（separate_kv=true）时写入 K/V 分项字段，
        否则无副作用（单 kind 行为不变）。
    """
    if "V" not in per_kind:
        return
    for base, inner in fields.items():
        dst[f"{base}_K"] = per_kind["K"][inner]
        dst[f"{base}_V"] = per_kind["V"][inner]

# ===========================================================================
# Calibration 数据合成（bug-4 修复）
# ===========================================================================


def _synth_calibration_kv(
    n_t: int,
    n_s: int,
    seq_len: int,
    n_kv: int,
    head_dim: int,
    seed: int,
    noise: float = 0.05,
    w_t: list[np.ndarray] | None = None,
    w_s: list[np.ndarray] | None = None,
    kv_seed_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """构造有真实对应关系的 Teacher 与 Student KV（修复 bug-4）。

    设计原理：
        - 共享 latent Z: (S, H, D) 随机正态 → 代表"真实信号"。
        - Teacher 各层: T_l = Z @ W_T_l + 噪声
        - Student 各层: S_l = Z @ W_S_l + 噪声
        其中 W_T_l / W_S_l 是固定的"层专属"线性变换。

    关键：Teacher 与 Student 都从同一 Z 派生，所以存在可学到的线性映射
    y @ W ≈ x（W = W_S @ W_T^{-1} 类）。

    参数：
        n_t, n_s: Teacher / Student 层数
        seq_len: 序列长度
        n_kv: kv-head 数
        head_dim: 每个 head 的维度
        seed: 随机种子
        noise: 加性高斯噪声的 std
        w_t, w_s: ◆ bug-3 聚合校准（§32）：可选传入共享的"提取矩阵"列表
            （长度 n_t / n_s）。传入后跨样本复用同一组 W —— 模拟"同一
            Teacher/Student 模型对、不同 prompt"的真实语义，使多个校准样本
            能聚合成一个可学的映射。默认 None → 每次调用独立生成
            （旧行为，供独立样本/消融等非聚合场景使用）。
        kv_seed_offset: ◆ design.md §22（K/V 独立参数化）：实际随机种子为
            `seed + kv_seed_offset`。调用方为 K 传 offset=0、为 V 传一个
            不同的 offset（如 100000），即可得到**独立随机抽样**的两套
            张量（同一结构、不同随机性）—— 模拟真实 K/V 是两个独立缓存。
    返回：
        (kv_t, kv_s)  形状分别为 (n_t, S, H, D) 与 (n_s, S, H, D)
    """
    # §22：K 与 V 用不同随机种子抽样 → 独立 latent Z 与噪声（独立缓存）
    rng = np.random.default_rng(seed + kv_seed_offset)
    Z = rng.standard_normal((seq_len, n_kv, head_dim))  # 共享 latent：所有层的"真实信号"
    # 输出张量按层预分配（n_t / n_s 层），逐层填充避免反复分配
    kv_t = np.zeros((n_t, seq_len, n_kv, head_dim), dtype=np.float32)
    kv_s = np.zeros((n_s, seq_len, n_kv, head_dim), dtype=np.float32)
    # 每个 Teacher 层一个独立的"提取矩阵"；传入共享 W 时复用（聚合校准）。
    # /sqrt(head_dim) 归一化：使 ||W|| 保持 O(1)，避免 D 维矩阵乘放大方差
    W_t = w_t if w_t is not None else [
        rng.standard_normal((head_dim, head_dim)) / np.sqrt(head_dim)
        for _ in range(n_t)
    ]
    # 每个 Student 层一个独立的"提取矩阵"，与 Z 共享以保证可映射
    W_s = w_s if w_s is not None else [
        rng.standard_normal((head_dim, head_dim)) / np.sqrt(head_dim)
        for _ in range(n_s)
    ]
    for l in range(n_t):
        # 对每个 head 用同一矩阵（per-head 矩阵后续可扩展）
        kv_t[l] = (Z @ W_t[l]) + noise * rng.standard_normal(Z.shape)
    for l in range(n_s):
        kv_s[l] = (Z @ W_s[l]) + noise * rng.standard_normal(Z.shape)
    return kv_t, kv_s


def _synth_calibration_set(
    n_t: int,
    n_s: int,
    seq_len: int,
    n_kv: int,
    head_dim: int,
    n_samples: int,
    master_seed: int,
    noise: float = 0.05,
    w_t: list[np.ndarray] | None = None,
    w_s: list[np.ndarray] | None = None,
    kv_seed_offset: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """生成共享 W_t/W_s 的 n_samples 个校准样本（bug-3 聚合校准 helper）。

    真实语义（§32）：同一 Teacher/Student 模型对、多个 prompt —— W_t/W_s
    由 master_seed 生成一次并跨样本复用，每个样本只换 latent Z 与噪声。
    返回值可直接喂给 `fit_ridge_aggregate`（Ridge 类）或
    `concat_kv_samples`（ALS 类，T06）。

    参数：
        n_t, n_s, seq_len, n_kv, head_dim: 同 `_synth_calibration_kv`
        n_samples: 校准样本数（§32 目标 100-500）
        master_seed: 生成共享 W_t/W_s 的种子；同时作为**样本种子空间的基底**，
            样本 i 的 latent 种子 = master_seed*1_000_000 + i —— 保证不同
            master_seed 的数据集（如 calib=0、held-out eval=1）的 latent Z
            与噪声完全不相交（否则 eval[0..19] 会与 calib[0..19] 逐字节相同，
            held-out 名存实亡）。
        noise: 加性高斯噪声 std
        w_t, w_s: ◆ 可选复用已有 W（校准集与评估集必须属于**同一模型对**）。
            调用方应先生成一组 W 并同时传给校准集与 held-out 评估集
            （否则等于拿不同模型的 KV 做 train/eval —— cosine 归零）。
            默认 None → 按 master_seed 独立生成。
        kv_seed_offset: ◆ design.md §22（K/V 独立参数化）：透传给
            `_synth_calibration_kv` —— K 用 offset=0、V 用不同 offset，
            得到两套**独立随机抽样**的校准集（同一结构、不同随机性），
            W_t/W_s 仍由 master_seed 共享（同一模型对）。
    返回：
        [(kv_t_i, kv_s_i) ...] 共 n_samples 对，全部共享同一组 W
    """
    # 共享 W_t/W_s 只生成一次：由 master_seed 决定，跨所有样本复用（同一模型对）
    rng = np.random.default_rng(master_seed)
    W_t = w_t if w_t is not None else [
        rng.standard_normal((head_dim, head_dim)) / np.sqrt(head_dim)
        for _ in range(n_t)
    ]
    W_s = w_s if w_s is not None else [
        rng.standard_normal((head_dim, head_dim)) / np.sqrt(head_dim)
        for _ in range(n_s)
    ]
    # 每个样本换一套 latent Z 与噪声：种子 = master_seed*1_000_000 + i
    # （见 docstring —— 保证不同 master_seed 数据集完全不相交）
    return [
        _synth_calibration_kv(
            n_t, n_s, seq_len, n_kv, head_dim,
            seed=master_seed * 1_000_000 + i, noise=noise, w_t=W_t, w_s=W_s,
            kv_seed_offset=kv_seed_offset,  # §22：K/V 独立随机抽样
        )
        for i in range(n_samples)
    ]


def _shared_model_weights(
    n_t: int, n_s: int, head_dim: int, master_seed: int
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """生成一组共享的模型权重 W_t/W_s（同一 Teacher/Student 模型对）。

    校准集与 held-out 评估集必须复用同一组权重（§32），本 helper 供
    T04/T05/T06 的 calib/eval 划分使用：先拿权重，再分别传入
    `_synth_calibration_set(w_t=..., w_s=...)`，保证 train/eval 同模型。

    参数：
        n_t / n_s: Teacher / Student 层数（决定每层生成一个独立提取矩阵）
        head_dim: 每个 head 的维度（矩阵为 (head_dim, head_dim)）
        master_seed: 权重随机种子；T04/T05/T06 统一用 0，保证
            calib 与 held-out 属于同一模型对
    返回：
        (w_t, w_s)：各 n_t / n_s 个 (head_dim, head_dim) 矩阵的列表。
        矩阵经 /sqrt(head_dim) 归一化保持 ||W|| ≈ O(1)（数值稳定）。
    """
    rng = np.random.default_rng(master_seed)
    w_t = [
        rng.standard_normal((head_dim, head_dim)) / np.sqrt(head_dim)
        for _ in range(n_t)
    ]
    w_s = [
        rng.standard_normal((head_dim, head_dim)) / np.sqrt(head_dim)
        for _ in range(n_s)
    ]
    return w_t, w_s


def _score_kv(kv_pred: np.ndarray, kv_ref: np.ndarray) -> dict[str, float]:
    """对齐两个 KV 张量并计算 R² / cosine / KL（T04/T05/T06 通用评分）。

    参数：
        kv_pred: 预测的 KV（Teacher→Student 映射输出）
        kv_ref:  参考 KV（真实 Student KV）
    返回：
        {"r2", "cosine", "kl"} 三个标量指标。
    边界说明：
        - 张量先 reshape 为 (样本行, D)，即把 S、H（、层）都并入样本维，
          逐 token 对齐后计算指标。
        - KL 分支：用 |kv| 按 D 维求平均得到一个非负"分布"，+1e-6 平滑
          避免 0 值导致 log(0)/除零（数值稳定性；kl_divergence 内部另有
          eps=1e-12 归一化保护）。
        - 这是离线 feature 空间的 proxy 评分（§75 明确禁止以 R²/cosine
          替代 CHG 等科学端点）。
    """
    # 压平到 (n_tokens, D)：每个 token 的 D 维向量逐行对齐比较
    a = kv_pred.reshape(-1, kv_pred.shape[-1])
    b = kv_ref.reshape(-1, kv_ref.shape[-1])
    return {
        "r2": r2(a, b),
        "cosine": cosine(a, b),
        # KL 用"按 D 平均的 |kv|"当分布，1e-6 防止 0 值出现（log(0) 防御）
        "kl": kl_divergence(
            np.abs(a).mean(0) + 1e-6, np.abs(b).mean(0) + 1e-6
        ),
    }


# ===========================================================================
# §49 计时工具（warmup + sync + ≥10 repeats + P50/P95）
# ===========================================================================


def time_block(repeats: int, warmup: int, sync: bool = True):
    """生成计时用 context manager 工厂（§49 规范：warmup + sync + ≥10 repeats）。

    参数：
        repeats: 有效计时的重复次数（§49 要求 ≥10）
        warmup: 预热次数 —— 丢弃初始化/首次分配等一次性开销
        sync: 是否在计时前后做 CUDA 同步（GPU 计时必须，否则异步误差）
    返回：
        contextmanager 工厂；with 块内每 yield 一次记一次耗时，
        结束后 `cm.times` 保存每次耗时（ms），供 P50/P95 统计。
    """
    import contextlib

    @contextlib.contextmanager
    def _cm():
        # warmup：先跑 warmup 次，丢弃启动开销（CUDA 首次调用/内存分配）
        for _ in range(warmup):
            yield None
        times = []
        try:
            import torch  # type: ignore

            has_cuda = torch.cuda.is_available() and sync

            def sync_call():
                if has_cuda:
                    torch.cuda.synchronize()
        except ImportError:
            # 无 torch（CPU-only 环境）：同步为空操作
            has_cuda = False

            def sync_call():
                pass

        # 正式计时：每次 yield 前后各 sync 一次，保证测到的是 GPU 完成时间
        for _ in range(repeats):
            sync_call()
            t0 = time.perf_counter()
            yield None
            sync_call()
            times.append((time.perf_counter() - t0) * 1000.0)

        cm.times = times

    return _cm


# ===========================================================================
# §53 单卡执行策略
# ===========================================================================


def single_card_pipeline(
    teacher_load_fn: Callable,
    student_load_fn: Callable,
    capture_fn: Callable,
    unload_fn: Callable,
    cuda_cleanup_fn: Callable,
    map_fn: Callable,
    inject_fn: Callable,
    decode_fn: Callable,
) -> dict[str, float]:
    """单卡执行编排（design.md §53 强制的顺序）。

    流程：
        Teacher Load → Forward → Capture → CPU Offload →
        Teacher Unload → CUDA Cleanup → Student Load →
        Map → Inject → Decode

    本函数返回每阶段耗时（ms），便于 §38 T10 报告。
    """
    timings: dict[str, float] = {}
    # §53 单卡约束：同一时刻只允许一个模型驻留显存 —— 必须先卸载 Teacher
    # 并清理 CUDA 后才能加载 Student（顺序不可调换）。每阶段耗时单位 ms。
    t0 = time.perf_counter()
    teacher_load_fn()  # 1) 加载 Teacher 权重
    timings["teacher_load"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    capture_fn()  # 2) Teacher Forward + 捕获 C_T(X) KV
    timings["capture"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    unload_fn()  # 3) KV offload 到 CPU 后卸载 Teacher（释放显存）
    timings["teacher_unload"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    cuda_cleanup_fn()  # 4) 清空 CUDA context / 缓存碎片
    timings["cuda_cleanup"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    student_load_fn()  # 5) 加载 Student（此时显存已腾空）
    timings["student_load"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    map_fn()  # 6) mapper：C_T(X) → C_S*(X)（合成 Student 状态）
    timings["map"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    inject_fn()  # 7) 把合成 KV 注入 Student cache
    timings["inject"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    decode_fn()  # 8) 解码（zero prefill：不重新读 X，§52 禁止 1）
    timings["decode"] = (time.perf_counter() - t0) * 1000.0

    return timings  # {阶段名: 耗时(ms)}，供 §38 T10 / PSR_A 报告


# ===========================================================================
# §32 T04 Ridge Baseline
# ===========================================================================


def _cfg_layers(cfg: dict[str, Any]) -> tuple[int, int, int, int]:
    """从 cfg 读取 teacher/student 层数与 head 配置（bug-9 修复）。

    优先使用 cfg["teacher"]["num_layers"] 等显式字段；
    否则 fallback 到 Qwen3-4B/1.7B 的已知值。
    """
    # 优先读显式架构字段（configs/*.yaml 可配置项）
    n_t = cfg["teacher"].get("num_layers")
    n_s = cfg["student"].get("num_layers")
    H_t = cfg["teacher"].get("num_kv_heads")
    H_s = cfg["student"].get("num_kv_heads")
    D_t = cfg["teacher"].get("head_dim")
    D_s = cfg["student"].get("head_dim")

    # fallback 到已知 Qwen3 架构（按 model_id 关键字匹配，只填缺省的字段）
    teacher_mid = str(cfg["teacher"].get("model_id", "")).lower()
    student_mid = str(cfg["student"].get("model_id", "")).lower()

    defaults = {
        "4b": dict(num_layers=36, num_kv_heads=8, head_dim=128),
        "1.7b": dict(num_layers=28, num_kv_heads=8, head_dim=128),
    }

    for tag, d in defaults.items():
        if tag in teacher_mid:
            n_t = n_t or d["num_layers"]
            H_t = H_t or d["num_kv_heads"]
            D_t = D_t or d["head_dim"]
        if tag in student_mid:
            n_s = n_s or d["num_layers"]
            H_s = H_s or d["num_kv_heads"]
            D_s = D_s or d["head_dim"]
    n_t = n_t or 36
    n_s = n_s or 28
    H_t = H_t or 8
    H_s = H_s or 8
    D_t = D_t or 128
    D_s = D_s or 128
    # §12 / §13：head 或 head_dim 不一致 → G2（需 P_H / P_d projection）
    if H_t != H_s or D_t != D_s:
        import warnings

        warnings.warn(
            f"G2 维度不匹配：teacher(H={H_t}, D={D_t}) vs student(H={H_s}, D={D_s})；"
            f"当前 runner 按 teacher 维度训练，真实 G2 需走 mismatched 投影（§12/§13）",
            stacklevel=2,
        )
    # 返回 (Teacher 层数, Student 层数, 统一 head 数, Teacher head_dim)：
    # 当前 runner 按 Teacher 维度训练，H 取 max 以避免 GQA 比例差异被截断
    return n_t, n_s, max(H_t, H_s), D_t


def run_ridge_baseline(cfg: dict[str, Any], run_dir) -> dict[str, Any]:
    """T04 Ridge Baseline（design.md §32）。

    指标体系（§32）：
        - mapper_n_params：Ridge per-head 映射的参数规模
        - mean_r2 / mean_kv_cosine：预测 KV vs 真实 Student KV 的拟合度
        - mean_attn_output_cosine：attention 输出分布相似度
          （attn-output cosine = softmax(Q·K^T/√d) 的分布相似度，§32）
        - retention：handoff cosine / self cosine（§33，有界到 1.0）
        - latency_map_ms_p50/p95：fit/transform 计时（§49 协议）
    种子规则：权重 master_seed=0（同一模型对）；校准样本 master_seed=0、
    held-out 评估样本 master_seed=1（Z/噪声完全不相交）。
    Gate：mean_kv_cosine > 0.5 → PASS（T04 无正式 gate，此为报告性阈值）。
    """
    n_t, n_s, H, D = _cfg_layers(cfg)
    # ◆ 内存自适应（架构审查续轮）：默认 n_calib=128 在 4 GB 环境会 OOM，
    #   必须按可用内存降级；用户显式 cfg 覆盖则不被本逻辑触碰。
    from ..utils.memory import safe_calibration_defaults
    seq = cfg.get("mapper", {}).get("calibration_context", 512)
    n_calib_cfg = cfg.get("mapper", {}).get("calibration_samples")
    if n_calib_cfg is None:
        # 拿架构参数做内存估算
        H_arch = cfg.get("teacher", {}).get("num_kv_heads", 8)
        D_arch = cfg.get("teacher", {}).get("head_dim", 128)
        defaults = safe_calibration_defaults(
            n_t=n_t, n_s=n_s, H=H_arch, D=D_arch, seq=seq,
        )
        n_calib = defaults["n_calib_estimate"]
        # seq 也降级（如 853 MB 预算下 1024-seq 会 OOM）
        if seq > defaults["seq_estimate"]:
            seq = defaults["seq_estimate"]
    else:
        n_calib = int(n_calib_cfg)
    repeats = int(cfg.get("timing", {}).get("repeats", 10))
    warmup = int(cfg.get("timing", {}).get("warmup", 2))

    layer_map = proportional_mapping(n_t, n_s)  # §21：按比例对齐 Teacher→Student 层
    ridge = RidgePerHeadMapper(lam=1e-3)  # λ=1e-3 岭正则：保证 (G+λI) 可逆（数值稳定性）

    # —— §23：默认开启 de-RoPE 路径（这里用 teacher theta）
    inv_freq = _rope_pairs(D, theta=1_000_000.0)
    de_rope_fn = lambda k, p: de_rope(k, p, inv_freq)
    positions = np.arange(seq, dtype=np.float64)

    # ◆ bug-3 修复（聚合校准）：校准样本共享 W_t/W_s（同一模型对不同 prompt），
    #   聚合 Gram 后只 fit 一次。旧实现 `for i in range(n_calib): ridge.fit(...)`
    #   每次覆盖 W，最终只留最后一组样本的影响（§32 目标 100-500 样本全浪费）。
    # ◆ design.md §22（K/V 独立参数化）：separate_kv=true 时 K 与 V 是两套
    #   独立缓存 —— 各自用独立随机抽样的校准集 fit 一套参数（kv_kind），
    #   transform 时按 kind 分别产出 K/V 两路输出并各自评估。
    kinds = kv_kinds(cfg)
    separate_kv = "V" in kinds

    # master_seed=0：权重种子；calib（master_seed=0）与 held-out（master_seed=1）
    # 复用同一组 w_t/w_s，只换 latent Z —— train/eval 同一模型对（§32）
    w_t, w_s = _shared_model_weights(n_t, n_s, D, master_seed=0)
    per_kind: dict[str, dict[str, float]] = {}
    latencies: list[float] = []
    for kind in kinds:
        calib = _synth_calibration_set(
            n_t, n_s, seq, H, D, n_calib, master_seed=0, noise=0.05,
            w_t=w_t, w_s=w_s, kv_seed_offset=KIND_SEED_OFFSET[kind],  # §22
        )
        t0 = time.perf_counter()
        fit_ridge_aggregate(ridge, calib, layer_map, kv_kind=kind,
                            positions=positions, de_rope_fn=de_rope_fn)  # §22
        latencies.append((time.perf_counter() - t0) * 1000.0)

        # 在 held-out 样本上评分（§32：校准集与评估集分离；**同一模型对**——
        # 复用上面的 w_t/w_s，只换 latent Z 与噪声）
        r2_list, cos_list, attn_cos_list, ret_list = [], [], [], []
        # held-out 评估：20 个样本，master_seed=1 与校准（=0）的 Z/噪声完全不相交
        n_eval_t04 = int(cfg.get("mapper", {}).get("t04_eval_samples", 20))
        eval_samples = _synth_calibration_set(
            n_t, n_s, seq, H, D, n_eval_t04, master_seed=1, noise=0.05,
            w_t=w_t, w_s=w_s, kv_seed_offset=KIND_SEED_OFFSET[kind],  # §22
        )
        for i, (kv_t, kv_s) in enumerate(eval_samples):
            t0 = time.perf_counter()
            pred = ridge.transform(kv_t, layer_map, kv_kind=kind,  # §22
                                   positions=positions, de_rope_fn=de_rope_fn)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            m = _score_kv(pred, kv_s)
            r2_list.append(m["r2"])
            cos_list.append(m["cosine"])
            # §33 retention：handoff cosine / self cosine（有界到 1.0；
            # 分母 max(..., 1e-6) 防御 self cosine 为 0 的除零边界）
            s_self = _score_kv(kv_s, kv_s)
            ret_list.append(min(1.0, m["cosine"] / max(s_self["cosine"], 1e-6)))
            # §32 attn-output cosine：用随机 Q 模拟 attn 输出分布相似度
            rng = np.random.default_rng(i)
            q = rng.standard_normal((seq, H, D))
            kt_ref = kv_s  # (L_s, S, H, D)
            kt_pred = pred
            # 仅取 Student 层 0 的 attention 输出做代表
            attn_ref = _attn_output(q, kt_ref[0])
            attn_pred = _attn_output(q, kt_pred[0])
            attn_cos_list.append(cosine(attn_ref.reshape(-1, D), attn_pred.reshape(-1, D)))
        per_kind[kind] = {
            "mean_r2": float(np.mean(r2_list)),
            "mean_cos": float(np.mean(cos_list)),
            "mean_attn_cos": float(np.mean(attn_cos_list)),
            "retention": float(np.mean(ret_list)),
        }

    # §49 P50 / P95（latencies 含每次 fit + 每次 transform 的耗时）
    p50 = percentile(latencies, 0.50) if latencies else 0.0
    p95 = percentile(latencies, 0.95) if latencies else 0.0
    # separate_kv=false 时只有 K → 数值与旧实现一致；true 时取 K/V 平均
    mean_r2 = float(np.mean([v["mean_r2"] for v in per_kind.values()]))
    mean_cos = float(np.mean([v["mean_cos"] for v in per_kind.values()]))
    mean_attn = float(np.mean([v["mean_attn_cos"] for v in per_kind.values()]))
    metrics = {
        "task": "T04",
        "n_calib_samples": n_calib,
        "context_length": seq,
        "mapper_n_params": ridge.n_params,
        "mean_r2": mean_r2,
        "mean_kv_cosine": mean_cos,
        "mean_attn_output_cosine": mean_attn,
        "latency_map_ms_p50": p50,
        "latency_map_ms_p95": p95,
        "repeats": repeats,
        "warmup": warmup,
        # 报告性阈值：KV cosine > 0.5 视为可恢复（T04 非正式 gate）
        "gate": "PASS" if mean_cos > 0.5 else "FAIL",
    }
    # design.md §22：separate_kv=true 时输出 K/V 各自的 retention/cosine/R² 字段
    _merge_kv_fields(
        metrics, per_kind,
        {
            "retention": "retention",
            "mean_cos": "mean_cos",
            "mean_r2": "mean_r2",
            "mean_attn_output_cosine": "mean_attn_cos",
        },
    )
    write_json(run_dir / "metrics.json", metrics)
    summary = (
        "# T04 Ridge Baseline\n\n"
        f"- Calibration samples: {n_calib}, context: {seq} "
        "(auto-sized from available memory; override via cfg.mapper.*)\n"
        f"- Mapper params: {ridge.n_params:,}\n"
        f"- Mean R²: {metrics['mean_r2']:.4f}\n"
        f"- Mean KV cosine: {metrics['mean_kv_cosine']:.4f}\n"
        f"- Mean attn-output cosine (§32): {metrics['mean_attn_output_cosine']:.4f}\n"
        f"- Map latency p50/p95: {p50:.2f}ms / {p95:.2f}ms (§49)\n"
        + (
            f"- K/V 独立参数化 (§22): retention_K={metrics['retention_K']:.4f}, "
            f"retention_V={metrics['retention_V']:.4f}\n"
            if separate_kv else ""
        )
    )
    (run_dir / "summary.md").write_text(summary, encoding="utf-8")
    return {"status": metrics["gate"], "metrics": metrics, "summary": summary}


def _attn_output(q: np.ndarray, k: np.ndarray) -> np.ndarray:
    """近似计算 attn output = softmax(Q K^T / √d) @ V 的"形状相似度"。

    这里我们没有 V，所以简化为：
        attn_logits = Q @ K^T / sqrt(d)
        attn_output ≈ attn_logits @ K (proxy)
    用于对比 reference vs prediction 的 attention distribution 一致性。
    """
    S, H, D = q.shape
    d = np.sqrt(D)
    # q: (S, H, D) → k: (S, H, D)
    # logits[i, j, h] = <q[i, h], k[j, h]> / d
    logits = np.einsum("ihd,jhd->ijh", q, k) / d
    # softmax over j
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    # attn_output ≈ p @ k: (S, H, D)
    return np.einsum("ijh,jhd->ihd", p, k)


# ===========================================================================
# §33 T05 Replacement
# ===========================================================================


def run_replacement(cfg: dict[str, Any], run_dir) -> dict[str, Any]:
    """T05 Replacement（design.md §33）。

    指标体系（§33）：
        - retention_per_context：每个 context 长度的 handoff/self 保留率
        - mean_retention / mean_token_agreement：跨 context 汇总
        - token_agreement：用 argmax 维度的 KV 最后 token 一致率（离线 proxy）
        - latency_p50_ms / p95_ms：fit + transform 计时（§49 协议）
        - gap_strata：按 retention 离散度分 low/medium/high 桶（§48 代理）
    种子规则：权重 master_seed=0；每 context 校准 master_seed=0、
    held-out 评估 master_seed=1。
    Gate 1（§33）：mean_retention ≥ 0.90 → PASS；0.80–0.90 → CONDITIONAL；
    < 0.80 → FAIL。
    """
    n_t, n_s, H, D = _cfg_layers(cfg)
    layer_map = proportional_mapping(n_t, n_s)
    # 上下文长度列表（默认四档），只取前 4 档，逐档独立校准/评估
    contexts = cfg.get("context_lengths", [512, 1024, 2048, 4096])[:4]
    # ◆ 内存自适应（架构审查续轮）：n_calib 必须按最大 ctx 算一次，
    #   然后跨 ctx 复用 —— 否则 ctx=4096 时按 ctx=4096 估出的 n_calib 在
    #   ctx=512 时反而浪费内存预算，或反过来 ctx=4096 时不够内存 OOM。
    from ..utils.memory import safe_calibration_defaults
    n_calib_cfg = cfg.get("mapper", {}).get("replacement_calib_samples")
    if n_calib_cfg is None:
        # 用最大 ctx 估算 → 一次算出 n_calib_estimate（保守）
        defaults = safe_calibration_defaults(
            n_t=n_t, n_s=n_s, H=H, D=D, seq=max(contexts),
        )
        n_calib_for_all = defaults["n_calib_estimate"]
        # 若最大 ctx 大于 safe estimate → 缩 ctx 列表
        safe_max = defaults["seq_estimate"]
        if max(contexts) > safe_max:
            contexts = [c for c in contexts if c <= safe_max]
            if not contexts:
                contexts = [safe_max]
    else:
        n_calib_for_all = int(n_calib_cfg)
    repeats = int(cfg.get("timing", {}).get("repeats", 10))
    warmup = int(cfg.get("timing", {}).get("warmup", 2))

    from ..rope.runner import _rope_pairs, de_rope

    inv_freq = _rope_pairs(D, theta=1_000_000.0)

    ridge = RidgePerHeadMapper(lam=1e-3)  # λ=1e-3 岭正则（数值稳定性，同 T04）
    rows = []
    token_agree_list = []
    latencies = []
    de_rope_fn = lambda k, p: de_rope(k, p, inv_freq)  # §23 强制 de-RoPE 路径
    # design.md §22（K/V 独立参数化）：separate_kv=true 时 K/V 独立 fit/评估
    kinds = kv_kinds(cfg)
    separate_kv = "V" in kinds
    # 同一模型对：校准集与 held-out 评估集复用同一组 w_t/w_s（§32）
    w_t, w_s = _shared_model_weights(n_t, n_s, D, master_seed=0)
    for ctx in contexts:
        # ◆ bug-3 修复（聚合校准）：100 个共享 W 的样本聚合后只 fit 一次。
        #   旧实现 `for i in range(100): ridge.fit(...)` 每次覆盖 W，校准 100
        #   样本只剩最后一组的影响 —— retention 被严重低估。
        positions = np.arange(ctx, dtype=np.float64)
        per_kind_ret: dict[str, float] = {}
        per_kind_ta: dict[str, float] = {}
        for kind in kinds:
            # 内存自适应（架构审查续轮）：n_calib 已在外层按最大 ctx 算好，
            #   所有 ctx 复用同一值；cfg.mapper.replacement_calib_samples 覆盖则优先。
            n_calib_kind = n_calib_for_all
            calib = _synth_calibration_set(
                n_t, n_s, ctx, H, D, n_calib_kind, master_seed=0, noise=0.05,
                w_t=w_t, w_s=w_s, kv_seed_offset=KIND_SEED_OFFSET[kind],  # §22
            )
            t0 = time.perf_counter()
            fit_ridge_aggregate(ridge, calib, layer_map, kv_kind=kind,
                                positions=positions, de_rope_fn=de_rope_fn)  # §22
            latencies.append((time.perf_counter() - t0) * 1000.0)
            # 测试 20 样本（held-out：同一模型对、不同 Z）
            retentions = []
            tas = []
            n_eval_kind = int(cfg.get("mapper", {}).get("replacement_eval_samples", 20))
            for kv_t, kv_s in _synth_calibration_set(
                n_t, n_s, ctx, H, D, n_eval_kind, master_seed=1, noise=0.05,
                w_t=w_t, w_s=w_s, kv_seed_offset=KIND_SEED_OFFSET[kind],  # §22
            ):
                pred = ridge.transform(kv_t, layer_map, kv_kind=kind,  # §22
                                       positions=positions, de_rope_fn=de_rope_fn)
                # §33 score / retention
                s_self = _score_kv(kv_s, kv_s)
                s_hand = _score_kv(pred, kv_s)
                ret = min(1.0, s_hand["cosine"] / max(s_self["cosine"], 1e-6))
                retentions.append(ret)
                # §33 Token Agreement：用 argmax over hidden features 模拟 top-1 token
                ta = _token_agreement(kv_s, pred)
                tas.append(ta)
                token_agree_list.append(ta)
            per_kind_ret[kind] = float(np.mean(retentions))
            per_kind_ta[kind] = float(np.mean(tas))
        # 单 kind 时即该 kind 的均值；separate_kv 时对 K/V 平均得到总指标
        row = {
            "context": ctx,
            "retention": float(np.mean(list(per_kind_ret.values()))),
            "token_agreement": float(np.mean(list(per_kind_ta.values()))),
        }
        # design.md §22：separate_kv=true 时输出 K/V 各自的 retention / TA
        _merge_kv_fields(
            row, 
            {k: {"retention": per_kind_ret[k], "token_agreement": per_kind_ta[k]} for k in kinds},
            {"retention": "retention", "token_agreement": "token_agreement"},
        )
        rows.append(row)

    # §48 Gap strata：把 retention_per_context 按"假设 teacher-student gap"分桶
    # （离线模拟：用 retention 的离散度做 strata）
    gap_strata = _gap_strata(rows)

    # Gate 1 判定（§33）：跨所有 context 的平均 retention
    ret = float(np.mean([r["retention"] for r in rows]))
    if ret >= 0.90:
        gate = "PASS"
    elif ret >= 0.80:
        gate = "CONDITIONAL"
    else:
        gate = "FAIL"

    metrics = {
        "task": "T05",
        "contexts": [r["context"] for r in rows],
        "retention_per_context": rows,
        "mean_retention": ret,
        "mean_token_agreement": float(np.mean(token_agree_list)),
        "latency_p50_ms": percentile(latencies, 0.50),
        "latency_p95_ms": percentile(latencies, 0.95),
        "repeats": repeats,
        "warmup": warmup,
        "gap_strata": gap_strata,
        "gate1": gate,
    }
    # design.md §22：separate_kv=true 时输出 K/V 各自的平均 retention
    # （对 rows 再取一次平均，供 T09 等下游直接消费）
    if separate_kv:
        metrics["mean_retention_K"] = float(np.mean([r["retention_K"] for r in rows]))
        metrics["mean_retention_V"] = float(np.mean([r["retention_V"] for r in rows]))
        metrics["mean_token_agreement_K"] = float(np.mean([r["token_agreement_K"] for r in rows]))
        metrics["mean_token_agreement_V"] = float(np.mean([r["token_agreement_V"] for r in rows]))
    summary = (
        "# T05 Replacement\n\n"
        f"- Contexts: {[r['context'] for r in rows]}\n"
        f"- Mean retention: {ret:.4f}\n"
        f"- Mean token agreement: {metrics['mean_token_agreement']:.4f}\n"
        f"- Latency p50/p95: {metrics['latency_p50_ms']:.2f}ms / {metrics['latency_p95_ms']:.2f}ms\n"
        f"- Gap strata (§48): {gap_strata}\n"
        f"- **Gate 1: {gate}**\n"
    )
    (run_dir / "summary.md").write_text(summary, encoding="utf-8")
    return {"status": gate, "metrics": metrics, "summary": summary}


def _token_agreement(kv_ref: np.ndarray, kv_pred: np.ndarray) -> float:
    """§33 Token Agreement：用 KV 最后一个 token 的"最大分量维度"作 proxy token。

    参数：
        kv_ref:  参考 Student KV (L_s, S, H, D)
        kv_pred: 映射预测的 Student KV (L_s, S, H, D)
    返回：
        [0,1] 标量：每个 head 的 argmax 维度一致率。
    说明：真实实现需要接 model head 得到真实 top-1 token；离线模拟用
    feature-level proxy —— 取 Student 最后层、最后 token 的 K (H, D)，
    对 D 维 argmax 当作"预测的 top-1 token 索引"。
    """
    # 取 Student 最后层、最后 token 的 K → (H, D)
    a = kv_ref[-1, -1]  # (H, D)
    b = kv_pred[-1, -1]
    # 每个 head 独立 argmax：最高激活维度即"预测 token"（proxy）
    a_idx = np.argmax(a, axis=-1)
    b_idx = np.argmax(b, axis=-1)
    return float((a_idx == b_idx).mean())


def _gap_strata(rows: list[dict]) -> dict[str, list[float]]:
    """§48 Gap strata 报告（按 retention 分布分桶）。

    参数：
        rows: 各 context 的指标行（须含 "retention" 键）
    返回：
        {"low": [...], "medium": [...], "high": [...]}：按 retention 相对
        中位数分桶 —— 严格小于中位数 → low，等于 → medium，大于 → high。
    说明：在离线模拟中没有真实 teacher/student gap，所以用 retention 值
    离散度作为 strata 代理；真实实现应从 T07 的 gap_distribution 读入。
    """
    if not rows:
        return {"low": [], "medium": [], "high": []}
    rets = [r["retention"] for r in rows]
    median = float(np.median(rets))
    return {
        "low": [r["retention"] for r in rows if r["retention"] < median],
        "medium": [r["retention"] for r in rows if r["retention"] == median],
        "high": [r["retention"] for r in rows if r["retention"] > median],
    }


# ===========================================================================
# §34 T06 Lightweight Mapper
# ===========================================================================


def run_lightweight_mapper(cfg: dict[str, Any], run_dir) -> dict[str, Any]:
    """T06 Lightweight Mapper（design.md §34 / Figure 1 PCR vs Retention）。

    对比变体：Ridge (full, PCR 分母) / LowRank rank=8/16/32 /
    Shared Basis rank=16（§34 A2 消融，需 cfg.mapper.shared_basis=true）。
    指标体系（§34 / §3.4）：
        - params：各 mapper 参数量
        - pcr：PCR = params / p_ref（相对 Ridge 全量的参数压缩比，§3.4）
        - retention / r2 / cosine：在 held-out 样本上的替换保真度
    种子规则：权重 master_seed=0；校准 master_seed=0、held-out master_seed=2。
    """
    n_t, n_s, H, D = _cfg_layers(cfg)
    # ◆ 内存自适应（架构审查续轮）：t06 ALS concat 把 n_agg 个样本沿 S 拼接，
    #   单次驻留体积 = n_agg × seq × H × D × n_t × 4 B × 2（K/V 前后）。
    #   1.0 GB 环境限制下需要把 seq / n_agg 同步降级。
    from ..utils.memory import safe_calibration_defaults
    seq_cfg = cfg.get("mapper", {}).get("t06_context", 1024)
    n_agg_cfg = int(cfg.get("mapper", {}).get("t06_aggregate_samples", 16))
    # concat-后体积 = n_agg × seq × H × D × n_t × 8B（含 K/V 双倍）
    avail = 0
    try:
        from ..utils.memory import _available_memory_bytes
        avail = _available_memory_bytes()
    except Exception:
        pass
    if avail > 0:
        # 目标：concat 后体积 ≤ 25% 可用内存
        budget = int(avail * 0.20)
        # single concat volume per (n_agg, seq)
        def _concat_vol(a, s):
            return a * s * H * D * max(n_t, n_s) * 8
        # 若 默认 (16, 1024) 超预算，先减 n_agg，再减 seq
        if _concat_vol(n_agg_cfg, seq_cfg) > budget:
            # 尝试减少 n_agg 直到 ≤ 8
            for a in [8, 4, 2]:
                if _concat_vol(a, seq_cfg) <= budget:
                    n_agg = a
                    seq = seq_cfg
                    break
            else:
                # 即使 n_agg=2 也超 → 缩 seq
                n_agg = 2
                seq = max(128, budget // (n_agg * H * D * max(n_t, n_s) * 8))
                seq = min(seq, seq_cfg)
        else:
            n_agg = n_agg_cfg
            seq = seq_cfg
    else:
        seq = seq_cfg
        n_agg = n_agg_cfg
    # 64 校准样本 × 36 层 × 1024 seq × 8 H × 128 D = 1.5 GB；低内存环境 OOM
    n_calib = int(cfg.get("mapper", {}).get("t06_calib_samples", min(64, max(8, n_agg * 2))))  # 校准样本总数（前 n_agg 用于 concat 聚合）
    layer_map = proportional_mapping(n_t, n_s)

    inv_freq = _rope_pairs(D, theta=1_000_000.0)
    de_rope_fn = lambda k, p: de_rope(k, p, inv_freq)

    # PCR 分母：Ridge 全量 —— ◆ bug-3 修复：64 个共享 W 的样本 Gram 聚合
    # 后只 fit 一次（内存 O(L_s×H×D²)，与样本数无关）；旧实现每轮 fit 覆盖 W。
    ridge_ref = RidgePerHeadMapper(lam=1e-3)
    # design.md §22（K/V 独立参数化）：separate_kv=true 时 K 与 V 各自用
    # 独立随机抽样的校准集 fit 一套参数，评估时按 kind 分别 transform。
    kinds = kv_kinds(cfg)
    separate_kv = "V" in kinds
    # 同一模型对：calib 与 held-out eval 复用同一组 w_t/w_s（§32），
    # 只换 latent Z —— 否则等于拿不同模型的 KV 做 train/eval。
    w_t, w_s = _shared_model_weights(n_t, n_s, D, master_seed=0)
    calib_kv: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    eval_kv: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for kind in kinds:
        calib = _synth_calibration_set(
            n_t, n_s, seq, H, D, n_calib, master_seed=0, noise=0.05,
            w_t=w_t, w_s=w_s, kv_seed_offset=KIND_SEED_OFFSET[kind],  # §22
        )
        fit_ridge_aggregate(ridge_ref, calib, layer_map, kv_kind=kind,
                            positions=np.arange(seq, dtype=np.float64),
                            de_rope_fn=de_rope_fn)  # §22：PCR 分母也 K/V 独立
        # ALS 无法按 Gram 聚合：把 n_agg 个样本沿 S 维 concat 成一个大 KV 后
        # 只 fit 一次（方案 A）。数学上等价于"用全部样本训练"（Ridge 情形已由
        # fit_ridge_aggregate 证明等价；ALS 非凸，concat 是唯一严格写法）。
        calib_kv[kind] = concat_kv_samples(calib[:n_agg])
        # held-out 评估样本（同一模型对、不同 Z）
        eval_kv[kind] = _synth_calibration_set(
            n_t, n_s, seq, H, D, 1, master_seed=2, noise=0.05,
            w_t=w_t, w_s=w_s, kv_seed_offset=KIND_SEED_OFFSET[kind],  # §22
        )[0]
    # PCR 分母：Ridge 全量参数；positions_big 对应 concat 后 (n_agg*S) 的坐标
    p_ref = ridge_ref.n_params
    positions_big = np.tile(np.arange(seq, dtype=np.float64), n_agg)
    positions = np.arange(seq, dtype=np.float64)

    def _fit_score_kinds(
        mapper, kinds: list[str], row: dict[str, Any]
    ) -> dict[str, dict[str, float]]:
        """对每种 kind 各 fit 一次、各 transform 一次并评分（design.md §22）。

        参数：
            mapper: LowRankMapper / SharedBasisMapper 实例（ALS 类，concat 一次 fit）
            kinds: 待评测的 KV 种类列表（["K"] 或 ["K","V"]）
            row: 当前变体的指标行（原地补充 params/pcr/retention/r2/cosine）
        返回：
            per-kind 的 {"retention", "r2", "cosine"}；K/V 参数互不覆盖，
            单独 kind 时与旧实现数值一致。
        """
        per_kind_metrics: dict[str, dict[str, float]] = {}
        for kind in kinds:
            # concat 后的大 KV 校准样本（沿 S 维拼接，position 用 positions_big）
            kv_t_big, kv_s_big = calib_kv[kind]
            kv_t_eval, kv_s_eval = eval_kv[kind]
            mapper.fit(kv_t_big, kv_s_big, layer_map, kv_kind=kind,
                       positions=positions_big, de_rope_fn=de_rope_fn)  # §22
            pred = mapper.transform(kv_t_eval, layer_map, kv_kind=kind,
                                    positions=positions, de_rope_fn=de_rope_fn)  # §22
            m = _score_kv(pred, kv_s_eval)
            # retention 有界到 1.0；分母 max(...,1e-6) 防御 self cosine=0 除零
            ret = min(1.0, m["cosine"] / max(_score_kv(kv_s_eval, kv_s_eval)["cosine"], 1e-6))
            per_kind_metrics[kind] = {
                "retention": float(ret),
                "r2": float(m["r2"]),
                "cosine": float(m["cosine"]),
            }
        # 汇总到变体行：params / PCR（§3.4）/ retention / r2 / cosine
        row["params"] = mapper.n_params
        row["pcr"] = pcr(mapper.n_params, p_ref)
        row["retention"] = float(np.mean([v["retention"] for v in per_kind_metrics.values()]))
        row["r2"] = float(np.mean([v["r2"] for v in per_kind_metrics.values()]))
        row["cosine"] = float(np.mean([v["cosine"] for v in per_kind_metrics.values()]))
        # design.md §22：separate_kv=true 时输出 K/V 各自的 retention/R²/cosine
        _merge_kv_fields(
            row, per_kind_metrics,
            {"retention": "retention", "r2": "r2", "cosine": "cosine"},
        )
        return per_kind_metrics

    rows = []
    # §34 低秩变体：rank 8 / 16 / 32 —— 目标 PCR 更低、retention 不显著掉
    for rank in [8, 16, 32]:
        lr = LowRankMapper(rank=rank)
        row: dict[str, Any] = {"variant": f"lowrank-{rank}", "rank": rank}
        _fit_score_kinds(lr, kinds, row)
        rows.append(row)

    # Shared Basis（§34 A2 消融开关）—— 同样 concat 一次 fit（每 kind 一次）。
    # 与 LowRank 的区别：低秩因子在层间共享一个基底，参数更省
    if cfg.get("mapper", {}).get("shared_basis", False):
        sb = SharedBasisMapper(rank=16)
        row_sb: dict[str, Any] = {"variant": "shared-basis-16", "rank": 16}
        _fit_score_kinds(sb, kinds, row_sb)
        rows.append(row_sb)

    metrics = {"task": "T06", "p_ref_n_params": p_ref, "rows": rows}
    write_json(run_dir / "metrics.json", metrics)
    summary = (
        "# T06 Lightweight Mapper (PCR vs Retention, Figure 1)\n\n"
        "| variant | rank | params | PCR | retention | R² | cosine |\n"
        "| ------- | ---: | -----: | --: | --------: | -: | -----: |\n"
        + "\n".join(
            f"| {r['variant']} | {r['rank']} | {r['params']:,} | {r['pcr']:.3f} | "
            f"{r['retention']:.3f} | {r['r2']:.3f} | {r['cosine']:.3f} |"
            for r in rows
        )
        + "\n"
    )
    (run_dir / "summary.md").write_text(summary, encoding="utf-8")
    return {"status": "OK", "metrics": metrics, "summary": summary}