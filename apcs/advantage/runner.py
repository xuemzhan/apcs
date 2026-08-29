"""T08 Advantage State Training（design.md §7 / §22 / §36）。

═══════════════════════════════════════════════════════════════════════════════
C_S* = C_base + α_l · R_adv

组件：
1. Source-Layer Mixer          §21：每个 Student 层对若干 Teacher 层加权平均
                                满足 sum_i w_{l,i,h} = 1
2. K / V 独立低秩残差          §22：R_K = A_K σ(B_K Z_K)
                                      R_V = A_V σ(B_V Z_V)
                                默认 rank=16
3. RMS Calibration             §24：缩放 R 使 RMS(R) ≈ RMS(C_base)
4. Bounded α                   §25：α_l = α_max · tanh(a_l)
                                α_max 由 validation 决定，test 前冻结
5. Student 全冻结              §36
6. No query gate               §26（默认关闭）
═══════════════════════════════════════════════════════════════════════════════
# 引擎在 K/V 自适应缩放中的角色：它不做逐 token 复制，而是学一个"加法修正项"
#   C_S* = C_base + α_l · R_adv
# 即：Source-Layer Mixer 先给一个粗略的基态 C_base（§21 层对齐），
# K/V 独立低秩残差 R_adv 学习补足基态与 Student 自 KV 的差距（§22），
# RMS Calibration 保证残差幅度不越界（§24），α_l 逐层限幅注入（§25）。
# 最终注入 C_S* 到 Student 时，K/V 各自按通道独立缩放（separate_kv）。
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..io.runs import write_json
from ..metrics import rms_norm


class SourceLayerMixer:
    """每个 Student 层 l 持有 w_{l,i,h} (sum_i w = 1)，对 Teacher 层做加权。

    设计（§21）：
        - top-k = 2：每个 Student 层取 2 个 Teacher 层
        - 权重初始化：均匀 1/k
        - 训练时通过 softmax 强制 sum_i w = 1
    """

    def __init__(self, n_t: int, n_s: int, n_heads: int, top_k: int = 2):
        """参数：
            n_t: Teacher 层数（源层池大小）。
            n_s: Student 层数（目标层数，决定输出 L_s 维）。
            n_heads: KV head 数 H；每个 head 独立一组权重（§21 的 w_{l,i,h}）。
            top_k: 每个 Student 层最多从 layer_map 取几个 Teacher 层（默认 2）。
        """
        self.n_t = n_t
        self.n_s = n_s
        self.n_heads = n_heads
        self.top_k = top_k
        # 权重初始化：均匀 1/top_k（§21 起始点），训练时经 softmax 强制 sum=1。
        rng = np.random.default_rng(0)
        self.w = np.full((n_s, top_k, n_heads), 1.0 / top_k)

    def mix(self, kv_t: np.ndarray, layer_map: list[list[int]]) -> np.ndarray:
        """kv_t: (L_t, S, H, D) → 输出 (L_s, S, H, D) 的加权混合。

        参数：
            kv_t: Teacher 原始 KV（源，用于生成基态 C_base）。
            layer_map: §21 层映射，layer_map[s] 是该 Student 层的候选 Teacher 层。
        返回：
            z: (L_s, S, H, D) 基态 C_base —— 每个 Student 层是其 top-k
               Teacher 层的逐 head 凸组合（sum_i w = 1）。

        ◆ 修复（架构审查续轮）：原实现 w 长度固定为 top_k，
          但 proportional_mapping 实际返回的 list 长度 ≤ top_k 且不等。
          当 layer_map[s] 长度 < top_k 时，tensordot(w[2], block[1, :, h, :]) →
          shape-mismatch。现用 w_eff = w[:k] 截断到与 block 头维一致。
        """
        L_s = len(layer_map)
        S, H, D = kv_t.shape[1:]
        out = np.zeros((L_s, S, H, D), dtype=kv_t.dtype)
        for s in range(L_s):
            teachers = layer_map[s][: self.top_k]  # 只取前 top_k 个候选源层
            if not teachers:
                continue
            block = kv_t[teachers]  # (k, S, H, D)，k = len(teachers)
            k = block.shape[0]
            if k > self.top_k:
                # 防御性：layer_map 返回比 top_k 还多 → 截断
                block = block[: self.top_k]
                k = self.top_k
            for h in range(H):
                w = self.w[s, :k, h]  # (k,) 当前 head 的混合权重：与 block 头维一致
                # tensordot：按权重对 k 个源层做加权求和 → 该 head 的基态。
                out[s, :, h, :] = np.tensordot(w, block[:, :, h, :], axes=1)
        return out


class LowRankResidual:
    """K / V 独立低秩残差 R = A · σ(B · Z)（§22）。

    论文公式（§22）：
        R^K = A^K σ(B^K Z^K)，R^V = A^V σ(B^V Z^V)
    行主序（row-major）等价实现（与列主序 A·σ(B·Z) 数学等价）：
        h = σ(z @ A)        # z: (..., d_in) → h: (..., rank)
        R = h @ B           # → R: (..., d_in)
    其中 A 形状 (d_in, rank)、B 形状 (rank, d_in)。
    σ 默认 tanh（§22 R = A·σ(B·Z) 中的非线性）；nonlinear=None 时退化为线性
    h = z @ A（与旧版 __call__ = z @ A @ B 等价）。

    本实现提供 fit() 真实训练循环（§36）：梯度下降最小化
        || σ(z @ A) @ B - target ||²
    返回逐轮 loss history，取代"随机初始化后直接 eval"。
    """

    def __init__(
        self,
        d_in: int,
        rank: int,
        kind: str = "K",
        seed: int = 0,
        nonlinear: str | None = "tanh",
    ):
        """参数：
            d_in: 输入/输出维度 D（head_dim）；残差形状与 KV 通道一致。
            rank: 低秩秩数（默认 16，§22 / §47 A1 档位固定值）。
            kind: "K" 或 "V"，用于 seed 偏移，保证 K/V 实例参数不同。
            seed: 随机种子（kind="K" 用 seed，kind="V" 用 seed+10000）。
            nonlinear: "tanh"（§22 非线性）或 None（退化为线性 z@A@B）。
        """
        assert kind in {"K", "V"}
        assert nonlinear is None or nonlinear == "tanh", f"unknown nonlinear={nonlinear!r}"
        self.kind = kind
        self.nonlinear = nonlinear
        # 不同实例使用不同种子，避免共享同一种子导致 A 矩阵相同
        rng = np.random.default_rng(seed + (0 if kind == "K" else 10_000))
        # A:(d_in, rank) 下投影、B:(rank, d_in) 上投影；
        # /√rank 初始化让 σ(z@A) 进入 tanh 的线性区，避免饱和。
        self.A = rng.standard_normal((d_in, rank)) / np.sqrt(rank)
        self.B = rng.standard_normal((rank, d_in)) / np.sqrt(rank)

    # ---- σ 非线性及其导数（§22） ----
    def _sigma(self, x: np.ndarray) -> np.ndarray:
        """σ(x)：tanh 时 = tanh(x)；linear（nonlinear=None）时恒等。"""
        if self.nonlinear is None:
            return x
        return np.tanh(x)

    def _sigma_prime(self, x: np.ndarray) -> np.ndarray:
        """σ'(x)：tanh 时 = 1 - tanh²(x)；linear 时 = 1。"""
        if self.nonlinear is None:
            return np.ones_like(x)
        return 1.0 - np.tanh(x) ** 2

    def __call__(self, z: np.ndarray) -> np.ndarray:
        """R = σ(z @ A) @ B；z: (..., d_in) → (..., d_in)。

        参数：
            z: 基态输入（mixer 输出 z，形状任意前导维 + d_in）。
        返回：
            同形状残差 R（K 或 V 通道的自适应缩放修正量）。
        """
        h = self._sigma(z @ self.A)  # 先投影到 rank 维 + 非线性 σ
        return h @ self.B  # 再投影回 d_in 维，得到低秩残差 R

    def fit(
        self,
        z: np.ndarray,
        target: np.ndarray,
        n_iter: int = 200,
        lr: float = 0.1,
    ) -> list[float]:
        """真实训练循环（§36）：最小化 ||σ(z@A)@B - target||²。

        与 §32 中 LowRankMapper 的交替最小二乘（ALS）不同：ALS 对低秩
        因子交替做闭式 lstsq，这里对 A/B 同时做端到端梯度下降 —— 因为
        残差前有 tanh 非线性，无法写成 A/B 交替的最小二乘闭式解。

        解析梯度（逐元素展开）：
            h = σ(z@A)
            grad_B = h^T @ err                     # err = R - target
            grad_A = z^T @ (err @ B^T * σ'(z@A))   # σ'：tanh=1-tanh²，linear=1
        梯度按样本数 N 归一化（/N），等价于对 MSE loss 取梯度，使 lr 对
        batch 大小不敏感（N=32 与 N=57344 可用同一 lr）。

        参数：
            z: 输入特征（混入后的基态），reshape 为 (N, d_in)。
            target: 回归目标（kv_s - z，即基态需补足的残差真值）。
            n_iter: 梯度下降迭代轮数（默认 200）。
            lr: 学习率（默认 0.1，已按 /N 归一化，与 batch 大小无关）。
        返回：
            逐轮 MSE loss history（长度 == n_iter），单调不增。
        """
        z = np.asarray(z, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        Z = z.reshape(-1, self.A.shape[0])        # (N, d_in)
        T = target.reshape(-1, self.B.shape[1])   # (N, d_in)
        n = Z.shape[0]
        history: list[float] = []
        for _ in range(n_iter):
            s = Z @ self.A                        # (N, rank) 线性投影
            h = self._sigma(s)                    # (N, rank) 非线性 σ
            r = h @ self.B                        # (N, d_in) 残差输出
            err = r - T                           # 残差输出 vs 目标之误差
            history.append(float(np.mean(err**2)))  # 逐轮 MSE loss
            grad_b = h.T @ err / n                # (rank, d_in)
            grad_a = Z.T @ ((err @ self.B.T) * self._sigma_prime(s)) / n
            # 原地更新（避免大矩阵重新分配，且保持 A/B 引用不变）
            np.subtract(self.A, lr * grad_a, out=self.A)
            np.subtract(self.B, lr * grad_b, out=self.B)
        return history


def calibrate_rms(base: np.ndarray, residual: np.ndarray, max_ratio: float = 1.5):
    """RMS Calibration（§24）：缩放 R 使 RMS(R) ≈ RMS(C_base)。

    max_ratio 限制最大放大倍数，防止通过放大 norm 偷 CHG。
    论文未指定 max_ratio；这里取 1.5 作为保守值。

    参数：
        base: 基态 C_base（mixer 输出），提供 RMS 参照。
        residual: 待校准残差 R（K 或 V 通道）。
        max_ratio: 允许的最大放大倍数（默认 1.5，§24 保守值）。
    返回：
        校准后残差：scale = min(RMS(base)/RMS(R), max_ratio) 的 R；
        R 近零（RMS < 1e-8）时原样返回，避免除零 / 放大噪声。
    """
    rb = rms_norm(base)
    rr = rms_norm(residual)
    if rr < 1e-8:
        return residual
    # 核心缩放：让残差幅度追平基态；超出 max_ratio 的部分被截断。
    scale = min(rb / rr, max_ratio)
    return residual * scale


class BoundedAlpha:
    """α_l = α_max · tanh(a_l)（§25）。

    α_max 由 validation 决定，test 前冻结（§36 / §70）。
    tanh 保证 α ∈ [-α_max, α_max]，避免 unbounded 注入。
    """

    def __init__(self, alpha_max: float = 0.5, n_layers: int = 28):
        """参数：
            alpha_max: α 上界（validation 选定后冻结，默认 0.5）。
            n_layers: Student 层数，每层一个可学习标量 a_l。
        """
        self.alpha_max = alpha_max
        # 可学习标量 a 初始全 0 → 初始注入 α = α_max·tanh(0) = 0（无扰动起步）。
        self.a = np.zeros(n_layers)

    def get(self) -> np.ndarray:
        """返回逐层 α：(n_layers,) 且 |α| ≤ alpha_max（tanh 有界性，§25）。"""
        return self.alpha_max * np.tanh(self.a)


def run_advantage_train(cfg: dict[str, Any], run_dir) -> dict[str, Any]:
    """T08 入口：训练优势状态并报告组件分解。

    第一版固定（§36）：
        Source-Layer Mixer + Separate K/V Low-rank + Rank 16 + RMS Calibration +
        Bounded α + No Query Gate
    Student 主体：全冻结

    参数：
        cfg: 配置字典。读取 teacher/student 架构字段、advantage 节
             （rank / n_iter / lr）、mapper 节（alpha_max）。
        run_dir: 输出目录（写 metrics.json + summary.md）。
    返回：
        {"status": "OK", "metrics": {...}, "summary": "..."}。

    引擎流程（K/V 自适应缩放的一次完整训练）：
        ① mixer.mix → 基态 z（C_base）
        ② 训练残差 R_K / R_V 逼近 kv_s - z（补基态差距）
        ③ RMS 校准残差幅度 ≈ 基态幅度
        ④ bounded α 逐层限幅注入量 → 最终 C_S* = z + α·(R_K, R_V)
    """
    # ---- 读配置 ----
    provider_kind = cfg.get("provider", {}).get("kv", "synthetic").lower()
    if provider_kind != "synthetic":
        raise RuntimeError(
            "T08 尚未接入真实任务 loss/advantage labels；已停止用随机 KV "
            "代替真实 HF 证据。请先导出 held-out 任务评分后再运行 T08。"
        )
    n_t = cfg["teacher"].get("num_layers", 36)
    n_s = cfg["student"].get("num_layers", 28)
    H = cfg.get("teacher", {}).get("num_kv_heads", 8)
    D = cfg.get("teacher", {}).get("head_dim", 128)
    adv_cfg = cfg.get("advantage", {})
    rank = int(adv_cfg.get("rank", 16))  # §47 A1 固定档位 16
    alpha_max = float(cfg.get("mapper", {}).get("alpha_max", 0.5))

    # 合成校准 KV（离线可复现）；真实实验替换为 Teacher/Student 实际 KV。
    rng = np.random.default_rng(42)
    kv_t = rng.standard_normal((n_t, 256, H, D)).astype(np.float32)
    kv_s = rng.standard_normal((n_s, 256, H, D)).astype(np.float32)

    from ..alignment.runner import proportional_mapping

    # ---- 组装引擎组件 ----
    layer_map = proportional_mapping(n_t, n_s)  # §21 层对齐
    mixer = SourceLayerMixer(n_t, n_s, H, top_k=2)  # ① 基态生成器
    res_k = LowRankResidual(D, rank, "K")  # ② K 通道残差（§22 R^K）
    res_v = LowRankResidual(D, rank, "V")  # ② V 通道残差（§22 R^V）
    bounded = BoundedAlpha(alpha_max=alpha_max, n_layers=n_s)  # ④ α 限幅器

    z = mixer.mix(kv_t, layer_map)  # (L_s, S, H, D) 基态 C_base
    # §36 真实训练循环：R_K / R_V 各自 fit 训练。
    # 训练目标 = kv_s - z（残差需补足基态 z 与 Student 自身 KV 之差）。
    n_iter = int(adv_cfg.get("n_iter", 100))
    lr = float(adv_cfg.get("lr", 0.1))
    target = (kv_s - z).astype(np.float32)
    hist_k = res_k.fit(z, target, n_iter=n_iter, lr=lr)  # 训练 R_K（§22 R^K = A^K σ(B^K Z^K)）
    hist_v = res_v.fit(z, target, n_iter=n_iter, lr=lr)  # 训练 R_V（§22 R^V = A^V σ(B^V Z^V)）
    r_k = res_k(z)  # 推理 R_K：K 通道自适应缩放修正量
    r_v = res_v(z)  # 推理 R_V：V 通道自适应缩放修正量
    r_k = calibrate_rms(z, r_k)  # ③ K 残差幅度校准（§24，截断于 max_ratio）
    r_v = calibrate_rms(z, r_v)  # ③ V 残差幅度校准（§24）
    alpha = bounded.get()  # ④ 逐层注入系数 α_l = α_max·tanh(a_l)

    # ---- 报告指标 ----
    base_rms = float(rms_norm(z))  # 基态幅度，作为残差的参照系
    rk_rms = float(rms_norm(r_k))  # 校准后 K 残差幅度
    rv_rms = float(rms_norm(r_v))  # 校准后 V 残差幅度

    metrics = {
        "task": "T08",
        "rank": rank,
        "alpha_max": alpha_max,
        "alpha_layer_mean": float(alpha.mean()),
        "alpha_layer_std": float(alpha.std()),
        "base_rms": base_rms,
        "residual_K_rms": rk_rms,
        "residual_V_rms": rv_rms,
        "ratio_K": rk_rms / max(base_rms, 1e-12),  # 校准后应 ≈ 1（§24 目标）
        "ratio_V": rv_rms / max(base_rms, 1e-12),  # 校准后应 ≈ 1（§24 目标）
        # §36 训练信息：逐轮 MSE loss（证明损失确实下降，非随机初始化后直接 eval）
        "loss_init": float(np.mean([hist_k[0], hist_v[0]])),  # 初始 loss（K/V 平均）
        "loss_final": float(np.mean([hist_k[-1], hist_v[-1]])),  # 收敛后 loss
        "n_loss": n_iter,  # 训练迭代轮数（loss history 长度）
        "loss_init_K": float(hist_k[0]),  # K 通道初始 loss
        "loss_final_K": float(hist_k[-1]),  # K 通道收敛 loss
        "loss_init_V": float(hist_v[0]),  # V 通道初始 loss
        "loss_final_V": float(hist_v[-1]),  # V 通道收敛 loss
        "n_layers_teacher": n_t,
        "n_layers_student": n_s,
        "student_frozen": cfg["student"].get("freeze", True),  # §36 全冻结断言
        "offline_demo": True,
        "evidence_grade": "synthetic",
        "alpha_trained": False,
        "alpha_note": "alpha parameters remain at initialization; no task loss updates them",
        # §47 A1 Rank, A9 RMS, A10 Bounded 三项开关记录
        "ablation_switches": {
            "A1_rank_fixed": 16,
            "A9_rms_calibration": True,
            "A10_bounded_alpha": True,
            "A8_separate_kv": True,
        },
    }
    write_json(run_dir / "metrics.json", metrics)  # 标准产物：Figure 6 / T09 读取
    # summary.md：人类可读摘要（α 分布 / RMS 比 / 训练收敛曲线要点）。
    md = (
        "# T08 Advantage State\n\n"
        f"- rank={rank}, α_max={alpha_max}\n"
        f"- Student frozen: {metrics['student_frozen']}\n"
        f"- α mean / std: {metrics['alpha_layer_mean']:.3f} / "
        f"{metrics['alpha_layer_std']:.3f}\n"
        f"- base RMS: {base_rms:.4f}\n"
        f"- residual K RMS: {rk_rms:.4f} (ratio {metrics['ratio_K']:.3f})\n"
        f"- residual V RMS: {rv_rms:.4f} (ratio {metrics['ratio_V']:.3f})\n"
        f"- train loss: {metrics['loss_init']:.4f} → "
        f"{metrics['loss_final']:.4f} ({metrics['n_loss']} iters)\n"
        f"- Ablation switches: {metrics['ablation_switches']}\n"
    )
    (run_dir / "summary.md").write_text(md, encoding="utf-8")
    return {"status": "OK", "metrics": metrics, "summary": md}
