"""Task-Aware Mapper: KV-level knowledge distillation（P1.1/P1.3 混合目标版）.

Reconstruction objective alone has a structural ceiling:
    min ||teacher_KV @ W - student_KV||²  ⇒  注入后学生最多复现"自己"
（mapper 目标 = 学生自 KV ⇒ 输出上限 = 学生自身处理 X 的结果）。

要突破天花板，必须把"教师优势"以学生注意力可读的形式保留在 V 里——
即直接优化任务目标。本实现采用**混合目标**（P1.1）：

    L = (1 - β) · L_recon(W)  +  β · L_task

    L_recon: 校准集上 ||mapped_KV - student_KV||²（稳定性锚，防漂移）
    L_task : 学生+注入KV 的答案分布 vs 目标分布
             （gold 标签 CE；无标注时用教师字母分布软目标 KD/KL）

参数化（P1.1，防小样本过拟合——默认只训练极少数任务参数）：
    task_delta_mode="diag": 冻结 ridge W，仅训练每 (kind,s,h) 的
        γ ∈ R^D（逐维缩放）与 b ∈ R^D（偏置），pred = (x@W)⊙γ + b
        —— 每头 2D 个参数，20 样本也可稳定估计
    task_delta_mode="full": 直接微调整个 W（原实现，参数 3.7M，需大数据）

逐层有界 α（P1.3）: 可选学习 per-layer α_s ∈ [0,1]（sigmoid 参数化），
    mapped_kv[s] *= α_s —— 支持"部分层注入"假说的可部署实现。

Two-phase training:
    Phase A: Standard Ridge (closed-form) for initial weights
    Phase B: Mixed-objective fine-tuning (gradient-based)
"""
from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


class TaskAwareRidgeMapper:
    """Ridge mapper with task-aware fine-tuning（混合目标 + 界参数化）.

    Phase A: KV reconstruction (closed-form Ridge)
    Phase B: (1-β)·recon + β·task fine-tuning（diag/full 参数化，可选逐层 α）
    """

    def __init__(
        self,
        lam: float = 1e-3,
        lr: float = 1e-3,
        n_finetune_steps: int = 50,
        finetune_batch_size: int = 4,
        task_loss_weight: float = 0.7,
        task_delta_mode: str = "diag",
        learn_layer_alpha: bool = False,
        task_lr: float | None = None,
        task_steps: int | None = None,
        task_batch_size: int | None = None,
    ):
        """Args:
            lam: Ridge L2 正则（Phase A）
            lr: Phase B 学习率（兼容旧名）
            n_finetune_steps: Phase B 步数（兼容旧名）
            finetune_batch_size: 批大小（兼容旧名）
            task_loss_weight: β ∈ [0,1] —— task 项权重（1-β 给 recon 锚）
            task_delta_mode: "diag"（默认，每头 2D 参数）| "full"（微调整个 W）
            learn_layer_alpha: 是否学习逐层有界 α（P1.3）
            task_lr/task_steps/task_batch_size: 新参数名（优先于旧名）
        """
        self.lam = lam
        self.lr = float(task_lr if task_lr is not None else lr)
        self.n_finetune_steps = int(task_steps if task_steps is not None else n_finetune_steps)
        self.finetune_batch_size = int(task_batch_size if task_batch_size is not None else finetune_batch_size)
        self.task_loss_weight = float(task_loss_weight)
        self.task_delta_mode = str(task_delta_mode)
        self.learn_layer_alpha = bool(learn_layer_alpha)

        # Phase A weights (numpy, frozen in diag mode)
        self.W: dict[tuple[str, int, int], np.ndarray] = {}
        # Phase B task deltas (torch): diag 模式的 γ/b；full 模式的 W 本体
        self._W_torch: dict[tuple[str, int, int], Any] = {}
        self._gamma: dict[tuple[str, int, int], Any] = {}
        self._beta: dict[tuple[str, int, int], Any] = {}
        self._layer_alpha: Any = None  # torch (L_s,) sigmoid 参数
        self._layer_map_ref: list[list[int]] | None = None
        self._fitted: set[str] = set()  # which kv_kinds have been fit

    @property
    def n_params(self) -> int:
        return int(sum(w.size for w in self.W.values()))

    # ------------------------------------------------------------------
    # Phase A：闭式 Ridge 初始化（与 RidgePerHeadMapper 同构，可被
    # fit_ridge_aggregate 直接填充 .W —— 变长真实校准安全）
    # ------------------------------------------------------------------

    def fit(
        self,
        kv_t: np.ndarray,
        kv_s: np.ndarray,
        layer_map: list[list[int]],
        kv_kind: str = "K",
        positions: np.ndarray | None = None,
        de_rope_fn: Callable | None = None,
    ):
        """Phase A: Standard Ridge fitting (closed-form)."""
        from .math import RidgePerHeadMapper

        ridge = RidgePerHeadMapper(lam=self.lam)
        ridge.fit(kv_t, kv_s, layer_map, kv_kind, positions, de_rope_fn)
        self.W.update(ridge.W)
        self._fitted.add(kv_kind)
        logger.info(
            "[TaskAware] Phase A complete for %s: n_params=%d",
            kv_kind, self.n_params,
        )

    # ------------------------------------------------------------------
    # Phase B：混合目标微调
    # ------------------------------------------------------------------

    def fit_task_aware(
        self,
        student_model: Any,
        student_tokenizer: Any,
        teacher_kv_cache: dict[str, Any],
        sample_rows: list[Any],
        layer_map: list[list[int]],
        kv_kind: str,
        teacher_answers: dict[str, str],
        device: str = "cuda:0",
        scoring_suffix: str = "\nAnswer:",
        default_choices: list[str] | None = None,
        teacher_letter_probs: dict[str, list[float]] | None = None,
        student_kv_calib: dict[str, tuple] | None = None,
        scoring_letter_ids: dict[str, int] | None = None,
    ):
        """Phase B: 混合目标微调（P1.1/P1.3）。

        每个校准样本：
        1. 用当前参数（可微）变换教师 KV
        2. 构建 DynamicCache → 学生 forward suffix → 字母分布
        3. L_task = CE(gold)（有标注）或 KL(student ‖ teacher_letters)（KD）
           L_recon = ||mapped_KV - student_KV||²（需 student_kv_calib）
        4. L = (1-β)·L_recon + β·L_task → 反传更新任务参数

        Args:
            teacher_letter_probs: {sample_id: [pA,pB,pC,pD]} 教师字母软分布（KD 用）
            student_kv_calib: {sample_id: (s_k, s_v)} 学生自 KV（recon 锚用）
            scoring_letter_ids: {letter: token_id}（评分字母表，须与学生一致）
        """
        import torch
        import torch.nn.functional as F

        # Phase A 完成性检查：fit_ridge_aggregate 直接写 .W（不经 self.fit），
        # 因此以 W 键为准，而非 _fitted 集合 —— 此前据此 Phase B 被静默跳过
        if not any(key[0] == kv_kind for key in self.W):
            logger.warning("[TaskAware] Phase A not done for %s, skipping Phase B", kv_kind)
            return

        device = torch.device(device)
        D = next(iter(self.W.values())).shape[0]
        self._layer_map_ref = layer_map
        L_s = len(layer_map)

        # ---- 构建可训练参数 ----
        task_params: list[torch.nn.Parameter] = []
        self._W_torch = {}
        self._gamma = {}
        self._beta = {}
        for key, w_np in self.W.items():
            kind, s, h = key
            if kind != kv_kind:
                continue
            w_t = torch.tensor(w_np, dtype=torch.float32, device=device)
            if self.task_delta_mode == "full":
                w_t.requires_grad_(True)
                task_params.append(w_t)
            else:  # diag: W 冻结，仅 γ/b 可训练
                gamma = torch.ones(D, dtype=torch.float32, device=device, requires_grad=True)
                beta = torch.zeros(D, dtype=torch.float32, device=device, requires_grad=True)
                self._gamma[key] = gamma
                self._beta[key] = beta
                task_params.extend([gamma, beta])
            self._W_torch[key] = w_t
        if not self._W_torch:
            logger.warning("[TaskAware] No weights found for %s", kv_kind)
            return

        # P1.3: 逐层有界 α（sigmoid 参数化；logit=+4 → α≈0.98 起步，
        # 避免 sigmoid(0)=0.5 把注入 KV 砍半导致 PPL 爆炸）
        if self.learn_layer_alpha:
            alpha_logit = torch.full(
                (L_s,), 4.0, dtype=torch.float32, device=device, requires_grad=True
            )
            self._layer_alpha = alpha_logit
            task_params.append(alpha_logit)

        if not task_params:
            logger.warning("[TaskAware] no trainable params for %s", kv_kind)
            return
        optimizer = torch.optim.Adam(task_params, lr=self.lr)

        # ---- 收集有效样本 ----
        valid_samples = []
        for row in sample_rows:
            if row.sample_id not in teacher_kv_cache:
                continue
            cached = teacher_kv_cache[row.sample_id]
            if cached is None:
                continue
            t_k_np, t_v_np, _, S_t = cached
            kv_np = t_k_np if kv_kind == "K" else t_v_np
            has_gold = row.answer is not None and str(row.answer).strip().upper() in "ABCD"
            has_soft = bool(teacher_letter_probs) and row.sample_id in teacher_letter_probs
            if not (has_gold or has_soft):
                continue
            if student_kv_calib is not None and row.sample_id not in student_kv_calib:
                # recon 锚要求学生自 KV；缺失则跳过（诚实：不做无锚训练）
                continue
            valid_samples.append((row, kv_np, S_t))

        if not valid_samples:
            logger.warning("[TaskAware] No valid samples for Phase B (%s)", kv_kind)
            return

        logger.info(
            "[TaskAware] Phase B (%s, mode=%s, β_task=%.2f): %d samples × %d steps",
            kv_kind, self.task_delta_mode, self.task_loss_weight,
            len(valid_samples), self.n_finetune_steps,
        )

        from ..rope.runner import _rope_pairs, de_rope
        inv_freq = _rope_pairs(D, theta=1_000_000.0)
        de_rope_fn = lambda k, p: de_rope(k, p, inv_freq)  # noqa: E731

        student_model.eval()
        rng = np.random.default_rng(0)
        for step in range(self.n_finetune_steps):
            idx = rng.choice(
                len(valid_samples),
                size=min(self.finetune_batch_size, len(valid_samples)),
                replace=False,
            )
            total_task = torch.tensor(0.0, device=device)
            total_recon = torch.tensor(0.0, device=device)
            n_task = 0

            for i in idx:
                row, kv_np, S_t = valid_samples[i]

                # 1. 可微变换（diag: 冻结 W + 可训练 γ/b；full: 可训练 W）
                mapped_kv = self._transform_differentiable(
                    kv_np, layer_map, kv_kind, D, device, de_rope_fn
                )

                # 2. recon 锚（P1.1 混合目标的稳定性项）
                if student_kv_calib is not None and row.sample_id in student_kv_calib:
                    s_pair = student_kv_calib[row.sample_id]
                    s_ref = s_pair[0] if kv_kind == "K" else s_pair[1]
                    min_S = min(mapped_kv.shape[1], s_ref.shape[1])
                    ref = torch.tensor(
                        np.asarray(s_ref)[:, :min_S], dtype=torch.float32, device=device
                    )
                    total_recon = total_recon + F.mse_loss(
                        mapped_kv[:, :min_S], ref
                    )

                # 3. task 损失
                cache = self._build_cache_differentiable(
                    mapped_kv, device, student_model.dtype
                )
                suffix_text = f"{row.query}\n{scoring_suffix}"
                suffix_ids = student_tokenizer(
                    suffix_text, return_tensors="np"
                ).input_ids.reshape(-1)
                n_query = len(suffix_ids)
                suffix_t = torch.as_tensor(
                    suffix_ids.reshape(1, -1), dtype=torch.long
                ).to(device)
                pos_ids = torch.arange(S_t, S_t + n_query, dtype=torch.long).unsqueeze(0).to(device)
                output = student_model(
                    suffix_t, past_key_values=cache, use_cache=False,
                    position_ids=pos_ids,
                )
                logits = output.logits[:, -1, :].float()
                del output, cache

                letter_ids_map = scoring_letter_ids or {
                    l: int(student_tokenizer.encode(l, add_special_tokens=False)[0])
                    for l in ["A", "B", "C", "D"]
                }
                letter_token_ids = torch.tensor(
                    [letter_ids_map[l] for l in ["A", "B", "C", "D"]],
                    dtype=torch.long, device=device,
                )
                letter_logits = logits[0, letter_token_ids]  # (4,)
                log_probs = F.log_softmax(letter_logits, dim=-1)

                gold = str(row.answer).strip().upper() if row.answer else None
                if gold in "ABCD" and gold:
                    total_task = total_task + F.nll_loss(
                        log_probs.unsqueeze(0),
                        torch.tensor(["ABCD".index(gold)], device=device),
                    )
                elif teacher_letter_probs and row.sample_id in teacher_letter_probs:
                    soft = torch.tensor(
                        teacher_letter_probs[row.sample_id], dtype=torch.float32, device=device
                    )
                    # KL(teacher ‖ student) —— 软目标蒸馏
                    total_task = total_task + F.kl_div(
                        log_probs.unsqueeze(0),
                        soft.unsqueeze(0),
                        log_target=False,
                        reduction="sum",
                    )
                else:
                    continue
                n_task += 1

            if n_task == 0:
                continue
            loss_task = total_task / n_task
            loss_recon = total_recon / len(idx)
            beta = max(0.0, min(1.0, self.task_loss_weight))
            loss = beta * loss_task + (1.0 - beta) * loss_recon

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step + 1) % 10 == 0:
                logger.info(
                    "[TaskAware] step %d/%d task=%.4f recon=%.4f",
                    step + 1, self.n_finetune_steps,
                    float(loss_task), float(loss_recon),
                )

        # ---- 回写 numpy ----
        if self.task_delta_mode == "full":
            for key, w_torch in self._W_torch.items():
                self.W[key] = w_torch.detach().cpu().numpy()
        else:
            self._gamma_np = {
                k: g.detach().cpu().numpy() for k, g in self._gamma.items()
            }
            self._beta_np = {
                k: b.detach().cpu().numpy() for k, b in self._beta.items()
            }
        if self.learn_layer_alpha and self._layer_alpha is not None:
            self._layer_alpha_np = torch.sigmoid(self._layer_alpha).detach().cpu().numpy()
        logger.info("[TaskAware] Phase B complete for %s", kv_kind)

    def _transform_differentiable(
        self,
        kv_np: np.ndarray,
        layer_map: list[list[int]],
        kv_kind: str,
        D: int,
        device: str,
        de_rope_fn: Callable,
    ) -> Any:
        """可微变换：de-RoPE(K) → per-head 线性 → (γ⊙ + b)（diag）→ 逐层 α。"""
        import torch

        L_t, S, H, _ = kv_np.shape
        L_s = len(layer_map)
        positions = np.arange(S, dtype=np.float64)

        if kv_kind == "K":
            kv_deroped_layers = [de_rope_fn(kv_np[l], positions) for l in range(L_t)]
            kv_t = torch.tensor(
                np.stack(kv_deroped_layers, axis=0), dtype=torch.float32, device=device
            )
        else:
            kv_t = torch.tensor(kv_np, dtype=torch.float32, device=device)

        out = torch.zeros(L_s, S, H, D, dtype=torch.float32, device=device)
        for s in range(L_s):
            teachers = layer_map[s] if s < len(layer_map) else [0]
            src = torch.stack(
                [kv_t[min(t, L_t - 1)] for t in teachers], dim=0
            ).mean(dim=0)  # (S, H, D)
            for h in range(H):
                key = (kv_kind, s, h)
                w = self._W_torch.get(key)
                if w is None:
                    continue
                base = torch.einsum("sd,dq->sq", src[:, h, :], w.detach()
                                   if self.task_delta_mode == "diag" else w)
                if self.task_delta_mode == "diag" and key in self._gamma:
                    base = base * self._gamma[key].unsqueeze(0) + self._beta[key].unsqueeze(0)
                out[:, :, h] = base

        # P1.3: 逐层有界 α
        if self.learn_layer_alpha and self._layer_alpha is not None:
            alpha = torch.sigmoid(self._layer_alpha).unsqueeze(1).unsqueeze(2).unsqueeze(3)
            out = out * alpha
        return out

    def _build_cache_differentiable(
        self,
        mapped_kv: Any,
        device: str,
        dtype: Any,
    ) -> Any:
        """Build DynamicCache from mapped KV (preserves gradients)."""
        from transformers import DynamicCache

        L_s, S, H, D = mapped_kv.shape
        cache = DynamicCache()
        for layer_idx in range(L_s):
            k = mapped_kv[layer_idx].permute(1, 0, 2).unsqueeze(0).to(dtype)
            v = mapped_kv[layer_idx].permute(1, 0, 2).unsqueeze(0).to(dtype)
            cache.update(k, v, layer_idx)
        return cache

    def transform(
        self,
        kv_t: np.ndarray,
        layer_map: list[list[int]],
        kv_kind: str = "K",
        positions: np.ndarray | None = None,
        de_rope_fn: Callable | None = None,
    ) -> np.ndarray:
        """Apply mapping（Phase A W + Phase B 任务增量 γ/b + 逐层 α）。"""
        from .math import RidgePerHeadMapper

        ridge = RidgePerHeadMapper(lam=self.lam)
        ridge.W = {k: v for k, v in self.W.items() if k[0] == kv_kind}
        out = ridge.transform(kv_t, layer_map, kv_kind, positions, de_rope_fn)

        gamma_np = getattr(self, "_gamma_np", None)
        if self.task_delta_mode == "diag" and gamma_np:
            for s in range(out.shape[0]):
                for h in range(out.shape[2]):
                    key = (kv_kind, s, h)
                    if key in gamma_np:
                        out[s, :, h, :] = (
                            out[s, :, h, :] * gamma_np[key] + self._beta_np[key]
                        )
        alpha_np = getattr(self, "_layer_alpha_np", None)
        if self.learn_layer_alpha and alpha_np is not None and len(alpha_np) == out.shape[0]:
            out = out * alpha_np[:, None, None, None]
        return out
