"""推理后端接口 + numpy 假模型 + torch 骨架（design.md §53 / §49）。

═══════════════════════════════════════════════════════════════════════════════
本模块对应 design.md：
    §53 单卡执行策略 —— Teacher Load → Forward → Capture → CPU Offload →
        Teacher Unload → CUDA Cleanup → Student Load → Map → Inject → Decode
    §52 禁止 1 —— Student 在真实 inference 阶段不重新读 X（zero prefill）
    §49 计时规范 —— sync() 在计时边界调用（GPU 后端 torch.cuda.synchronize）
    §75 诚实性 —— 骨架接口必须显式 raise，不得静默跑 mock

后端契约（可替换接口 numpy → torch）：
    - load_model(run): 按 run 参数构造模型；参数缺失必须显式 raise。
    - forward_prefill(model, tokens): 返回 past_key_values（numpy 后端为
      (L_t, S, H, D) ndarray）。
    - decode(model, tokens, past_key_values): 返回 (token_out, new_pkv)。
      只消费注入的 KV 状态与当前 token，绝不接触 teacher 原始 token 序列。
    - inject(model, kv): 把映射后 (L_s, S, H, D) KV 注入 Student 缓存。
    - unload(model): 释放模型（GPU 后端还 CPU offload）。
    - sync(): §49 计时边界同步；CPU 后端 no-op。
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# 后端抽象接口
# ---------------------------------------------------------------------------


class InferenceBackend:
    """可替换推理后端接口（§53；numpy → torch）。

    骨架约定：基类方法默认显式 raise NotImplementedError，具体后端按需覆盖；
    任何分支都不得静默返回 mock 结果（§75 诚实性）。
    """

    name: str = "abstract"

    def _not_impl(self, method: str) -> NotImplementedError:
        """生成显式未实现异常（骨架统一出口）。"""
        return NotImplementedError(
            f"后端 {self.name} 未实现 {method}()：推理后端为接口骨架（§53），需 numpy / torch 具体实现"
        )

    def load_model(self, run: dict[str, Any]) -> object:
        """加载模型（§53 Teacher/Student Load）。run 为单侧模型参数。"""
        raise self._not_impl("load_model")

    def forward_prefill(self, model, tokens) -> object:
        """Teacher Forward + Capture：返回 past_key_values（§53）。

        返回类型为后端相关：numpy 后端是 (L_t, S, H, D) ndarray；
        torch 后端是 PKV 结构（tuple of tensors / DynamicCache）。
        """
        raise self._not_impl("forward_prefill")

    def decode(self, model, tokens, past_key_values) -> tuple[Any, Any]:
        """单步 decode：返回 (token_out, new_pkv)（§53 Decode）。

        §52 禁止 1：本方法只允许消费注入的 past_key_values 与当前 token，
        不得重新读取 Teacher 的 token 序列（zero prefill）。
        """
        raise self._not_impl("decode")

    def inject(self, model, kv) -> object:
        """把映射后 KV 注入 Student 缓存并返回可消费的 PKV（§53 Inject）。"""
        raise self._not_impl("inject")

    def unload(self, model) -> None:
        """释放模型（§53 Teacher Unload；GPU 后端含 CPU offload）。"""
        raise self._not_impl("unload")

    def sync(self) -> None:
        """§49 计时边界同步（CPU 后端 no-op；GPU 后端 torch.cuda.synchronize）。"""
        return None


# ---------------------------------------------------------------------------
# numpy 小型假模型（§52 可观测性载体）
# ---------------------------------------------------------------------------


class NumpyFakeModel:
    """numpy 小型假模型：2 层 decoder，KV 缓存只与 token embedding 相关。

    设计动机（§52 禁止 1 的可验证性）：
        假模型的 KV 是 token 的确定性函数：
            kv[l, s, h] = Wkv[l, h] @ embed(tok_s)
        因此 capture 得到的 KV 张量携带了 teacher 的全部上下文信息；
        decode 只需要消费注入的 KV 张量 + 当前 token，结构上就
        "不可能"重新读取 teacher 的 token 序列 —— 配合计数器
        (prefill_calls / decode_calls) 提供 §52 的统计断言。

    计数器契约（管线统计断言依赖）：
        - forward_prefill 每调用一次 prefill_calls += 1（Student 必须为 0）
        - decode 每调用一次 decode_calls += 1（必须 == n_gen）
    """

    def __init__(
        self,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        vocab: int,
        rng: np.random.Generator,
    ) -> None:
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.vocab = vocab
        # 固定随机权重（seed 由后端从 run 参数解析）
        self.E = (rng.standard_normal((vocab, head_dim)) * 0.02).astype(np.float32)
        self.Wkv = (
            rng.standard_normal((n_layers, n_kv_heads, head_dim, head_dim))
            / np.sqrt(head_dim)
        ).astype(np.float32)
        self.Wout = (
            rng.standard_normal((n_layers * n_kv_heads * head_dim, vocab))
            / np.sqrt(n_layers * n_kv_heads * head_dim)
        ).astype(np.float32)
        # §52 可观测计数器
        self.prefill_calls = 0
        self.decode_calls = 0
        # inject 后挂载的 KV 状态（供测试检查）
        self.injected_kv: np.ndarray | None = None

    def embed(self, tokens: np.ndarray) -> np.ndarray:
        """token ids → embedding（(..., D)）。"""
        return self.E[np.asarray(tokens)]

    def next_kv(self, tokens: np.ndarray) -> np.ndarray:
        """按"KV 只与 token embedding 相关"计算 (..., L, H, D) 新 KV。"""
        emb = self.embed(tokens)  # (..., D)
        return np.einsum("lhdd,...d->...lhd", self.Wkv, emb)

    def logits(self, kv: np.ndarray) -> np.ndarray:
        """由最新一行 KV 计算 vocab logits（(vocab,)）。"""
        last = np.asarray(kv)[:, -1]  # (L, H, D)
        return self.Wout.T @ last.reshape(-1)


# ---------------------------------------------------------------------------
# numpy 后端（完整可运行，供测试 / CI）
# ---------------------------------------------------------------------------


class NumpyBackend(InferenceBackend):
    """numpy 后端：完整实现 §53 链路，供测试与 CI（offline demo）。

    §75 诚实性：模型参数缺失时显式 raise NotImplementedError，
    绝不静默使用默认 mock。
    """

    name: str = "numpy"

    # run 参数必需键（缺任一 → 显式 raise）
    _REQUIRED: tuple[str, ...] = ("num_layers", "num_kv_heads", "head_dim", "vocab_size")

    def load_model(self, run: dict[str, Any]) -> NumpyFakeModel:
        """按 run 参数构造 NumpyFakeModel（§53 Teacher/Student Load）。

        参数缺失 → 显式 raise（不得静默跑默认 mock，§75）。
        """
        missing = [k for k in self._REQUIRED if run.get(k) is None]
        if missing:
            raise NotImplementedError(
                f"numpy 后端缺少模型参数 {missing}（role={run.get('role')}）：骨架要求显式传参，禁止静默使用默认 mock（§75）"
            )
        rng = np.random.default_rng(int(run.get("seed", 0)))
        return NumpyFakeModel(
            n_layers=int(run["num_layers"]),
            n_kv_heads=int(run["num_kv_heads"]),
            head_dim=int(run["head_dim"]),
            vocab=int(run["vocab_size"]),
            rng=rng,
        )

    def forward_prefill(self, model: NumpyFakeModel, tokens) -> np.ndarray:
        """§53 Forward + Capture：返回 (L_t, S, H, D) KV 张量（已在主机内存）。"""
        model.prefill_calls += 1
        kv = model.next_kv(np.asarray(tokens).reshape(-1)).transpose(1, 0, 2, 3)
        # einsum 输出 (S, L, H, D) → 转置为仓库约定 (L, S, H, D)
        return np.ascontiguousarray(kv)

    def decode(self, model: NumpyFakeModel, tokens, past_key_values) -> tuple[Any, Any]:
        """单步 decode：追加一行 KV，输出 (next_token, new_pkv)（§53）。

        §52 禁止 1：只消费 past_key_values 与当前 token —— 不读 teacher tokens。
        """
        model.decode_calls += 1
        tok = int(np.asarray(tokens).reshape(-1)[0])
        new_row = model.next_kv(np.array([tok]))[0]  # (L, H, D)
        pkv = np.concatenate(
            [np.asarray(past_key_values), new_row[:, None, :, :]], axis=1
        )
        token_out = int(np.argmax(model.logits(pkv)))
        return token_out, pkv

    def inject(self, model: NumpyFakeModel, kv) -> object:
        """§53 Inject：形状校验后把 (L_s, S, H, D) KV 挂到 Student 并返回。

        校验失败显式 raise，不静默降级（§52 禁止 8 的精神）。
        """
        kv = np.asarray(kv)
        expected = (model.n_layers, model.n_kv_heads, model.head_dim)
        if kv.ndim != 4 or kv.shape[0] != expected[0]:
            raise NotImplementedError(
                f"注入 KV 形状 {kv.shape} 与 Student {expected} 不匹配：inject 必须显式校验，禁止静默 re-prefill（§52 禁止 8）"
            )
        model.injected_kv = kv
        return kv

    def unload(self, model: NumpyFakeModel) -> None:
        """§53 Teacher Unload：numpy 无显存，仅清引用 + 归零计数器。"""
        model.prefill_calls = 0
        model.decode_calls = 0


# ---------------------------------------------------------------------------
# torch 后端骨架（真实 GPU 路径，未实现）
# ---------------------------------------------------------------------------


class TorchBackend(InferenceBackend):
    """torch 后端接口骨架（§53 真实 GPU 路径）。

    只定义接线，不实现 CI 无法验证的大型 GPU forward：
        - load_model      → transformers HF 模型（Teacher / Student）
        - forward_prefill → 真实 attention + KV cache capture
        - decode          → 增量 decode（消费注入 KV）
        - inject          → 把映射后 KV 写入 Student 显存缓存
        - sync            → torch.cuda.synchronize()（§49）

    骨架阶段一律显式 raise NotImplementedError，禁止静默 mock（§75 诚实性）。
    真实实现以 numpy 后端为行为参考，接入后即可端到端替换。
    """

    name: str = "torch"

    def load_model(self, run: dict[str, Any]) -> object:
        """§53 Teacher/Student Load：真实 HF 模型加载未实现，显式 raise。

        torch 分支 import 放函数内 try：CPU/CI 环境无 torch 也可安全导入本模块。
        """
        try:
            import torch  # type: ignore

            _ = torch.cuda.is_available()
        except ImportError:
            pass
        raise NotImplementedError(
            "torch 后端为接口骨架：真实 GPU 推理需要 transformers + CUDA（§53）。本骨架不实现无法验证的大型 forward；numpy 后端用于测试/CI。"
        )
