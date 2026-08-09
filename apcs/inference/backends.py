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

    后端注册/调度方式：
        每个后端以 name 字段标识（"numpy" / "torch" / "abstract"），
        管线不感知具体实现，只按统一契约调用方法 —— 这就是"注册/调度"：
        选择哪个后端在管线构造时决定（HandoffPipeline(backend, ...)），
        之后所有阶段（Load / Forward / Capture / Decode / Inject）全部
        通过本接口分发到具体后端。

    各后端与 KV 缓存的交互方式（契约核心）：
        - load_model(run)：构造模型；模型自身不持有 KV，KV 由下面方法进出。
        - forward_prefill(model, tokens)：Teacher 侧 Prefill，产出完整
          past_key_values（numpy 为 (L_t, S, H, D) ndarray）——这是
          "Teacher 自产 KV 缓存"的 Capture 出口。
        - decode(model, tokens, past_key_values)：增量 decode，只复用
          注入/传入的 KV 缓存 + 当前 token，输出新 token 与扩长后的 PKV。
        - inject(model, kv)：把映射后的 KV 写入 Student 的缓存结构并返回
          可消费的 PKV —— 这是"缓存复用"的入口，Student 不再自行 prefill。
    """

    name: str = "abstract"

    def _not_impl(self, method: str) -> NotImplementedError:
        """生成显式未实现异常（骨架统一出口）。

        消息携带后端 name 与缺失的方法名，便于快速定位是哪个后端
        缺了哪个阶段（§75 诚实性：接口骨架必须显式报错，禁止静默 mock）。
        """
        return NotImplementedError(
            f"后端 {self.name} 未实现 {method}()：推理后端为接口骨架（§53），需 numpy / torch 具体实现"
        )

    def load_model(self, run: dict[str, Any]) -> object:
        """加载模型（§53 Teacher/Student Load）。

        run 为单侧模型参数 dict（管线按 role 分别为 teacher / student
        各构造一份）；后端据此解析 num_layers / num_kv_heads / head_dim /
        vocab_size 等显式参数。run 参数检查：关键参数缺失必须显式 raise
        （§75），不得静默使用默认 mock；torch 后端则加载真实 HF 模型。
        """
        raise self._not_impl("load_model")

    def forward_prefill(self, model, tokens) -> object:
        """Teacher Forward + Capture：返回 past_key_values（§53）。

        返回类型为后端相关：numpy 后端是 (L_t, S, H, D) ndarray；
        torch 后端是 PKV 结构（tuple of tensors / DynamicCache）。
        Teacher 的 KV 缓存在这里生成并整体返回（已在主机内存，
        即 CPU Offload 的产物），后续仅作只读传给 Map / Inject。
        """
        raise self._not_impl("forward_prefill")

    def decode(self, model, tokens, past_key_values) -> tuple[Any, Any]:
        """单步 decode：返回 (token_out, new_pkv)（§53 Decode）。

        缓存复用条件：本方法只复用注入的 past_key_values 与当前 token，
        把当前 token 的 KV 一行追加到序列维（拼接），返回扩长后的
        new_pkv 作为下一步的输入 —— 全程不重新读输入序列。
        §52 禁止 1：不得重新读取 Teacher 的 token 序列（zero prefill）。
        """
        raise self._not_impl("decode")

    def inject(self, model, kv) -> object:
        """把映射后 KV 注入 Student 缓存并返回可消费的 PKV（§53 Inject）。

        缓存复用条件：只有 inject 成功之后 decode 才被允许 —— Student
        侧不再调用 forward_prefill；因此注入失败（形状不匹配等）必须
        显式 raise，禁止静默降级成 re-prefill（§52 禁止 8）。
        """
        raise self._not_impl("inject")

    def unload(self, model) -> None:
        """释放模型（§53 Teacher Unload；GPU 后端含 CPU offload）。

        释放的是模型权重与显存占用，KV 缓存产物已在主机内存，不受影响；
        这样后续 Student 阶段可复用同一进程而不冲突。
        """
        raise self._not_impl("unload")

    def sync(self) -> None:
        """§49 计时边界同步（CPU 后端 no-op；GPU 后端 torch.cuda.synchronize）。

        所有计时阶段（见 pipeline._stage）在进入/退出时都调用 sync，
        保证 GPU 异步 kernel 完成后再读墙钟，计时才可比较。
        """
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

    KV 缓存结构（与 numpy 后端契约一致）：
        - next_kv(tokens) 输出 (..., L, H, D)：每个 token 位置一段 KV。
        - 缓存形状约定 (L, S, H, D)：L = 层数（n_layers），
          S = 序列长度（token 数），H = n_kv_heads，D = head_dim。
        - decode 时"复用"上一轮 PKV，把当前 token 的新行 KV 追加到
          序列维 S 上（拼接），序列长度逐 token 增长。

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
        # E: (vocab, D) token embedding 表 —— embed 按 token id 查表
        self.E = (rng.standard_normal((vocab, head_dim)) * 0.02).astype(np.float32)
        # Wkv: (L, H, D, D) 每 (层, KV头) 一个线性变换 —— 决定 KV 与 token 的关系
        self.Wkv = (
            rng.standard_normal((n_layers, n_kv_heads, head_dim, head_dim))
            / np.sqrt(head_dim)
        ).astype(np.float32)
        # Wout: (L*H*D, vocab) 输出投影 —— 由最新一行 KV 计算 logits
        self.Wout = (
            rng.standard_normal((n_layers * n_kv_heads * head_dim, vocab))
            / np.sqrt(n_layers * n_kv_heads * head_dim)
        ).astype(np.float32)
        # §52 可观测计数器
        self.prefill_calls = 0
        self.decode_calls = 0
        # inject 后挂载的 KV 状态（供测试检查；模型本身不持有缓存，
        # KV 一律以显式参数进出 —— 这保证缓存状态可复用、可审计）
        self.injected_kv: np.ndarray | None = None

    def embed(self, tokens: np.ndarray) -> np.ndarray:
        """token ids → embedding（(..., D)）。

        边界：token id 必须落在 [0, vocab) 内，越界会触发 numpy
        IndexError —— 这与真实 HF 模型的 tokenizer 边界一致。
        """
        return self.E[np.asarray(tokens)]

    def next_kv(self, tokens: np.ndarray) -> np.ndarray:
        """按"KV 只与 token embedding 相关"计算 (..., L, H, D) 新 KV。

        einsum "lhdd,...d->...lhd"：对每个 (层, KV头) 用 Wkv 线性变换
        embedding，输出形状 (..., L, H, D)；末尾的 D 维即该 token 的
        新一行 KV（后续 decode 用它拼接进序列维）。
        """
        emb = self.embed(tokens)  # (..., D)
        return np.einsum("lhdd,...d->...lhd", self.Wkv, emb)

    def logits(self, kv: np.ndarray) -> np.ndarray:
        """由最新一行 KV 计算 vocab logits（(vocab,)）。

        边界（数据流）：输入 kv 形状 (L, S, H, D)，取序列维最后一列
        [:, -1] 作为"当前最新 token"的 KV；因此要求 S >= 1，
        空 KV（S == 0）会直接越界 —— 调用方必须保证注入后再 decode。
        """
        last = np.asarray(kv)[:, -1]  # (L, H, D)
        return self.Wout.T @ last.reshape(-1)


# ---------------------------------------------------------------------------
# numpy 后端（完整可运行，供测试 / CI）
# ---------------------------------------------------------------------------


class NumpyBackend(InferenceBackend):
    """numpy 后端：完整实现 §53 链路，供测试与 CI（offline demo）。

    与 KV 缓存的交互：
        - forward_prefill：Teacher 侧整体计算并返回 (L_t, S, H, D) KV。
        - inject：形状校验后把映射 KV 挂到 Student 的 injected_kv。
        - decode：每步复用上一步 PKV + 追加当前 token 一行（序列维拼接），
          模拟真实推理里"KV 缓存逐 token 增长、无需重算历史"的行为。

    §75 诚实性：模型参数缺失时显式 raise NotImplementedError，
    绝不静默使用默认 mock。
    """

    name: str = "numpy"

    # run 参数必需键（缺任一 → 显式 raise，§75）
    # 检查逻辑见 load_model：对 run.get(k) is None 的键集合报错
    _REQUIRED: tuple[str, ...] = ("num_layers", "num_kv_heads", "head_dim", "vocab_size")

    def load_model(self, run: dict[str, Any]) -> NumpyFakeModel:
        """按 run 参数构造 NumpyFakeModel（§53 Teacher/Student Load）。

        参数缺失 → 显式 raise（不得静默跑默认 mock，§75）。
        边界（run 参数检查）：只校验必需键是否存在（值非 None），
        缺了任意一个就整体拒绝构造，宁可失败也不给默认值；
        seed 取 run.get("seed", 0)，保证同配置可复现。
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
        """§53 Forward + Capture：返回 (L_t, S, H, D) KV 张量（已在主机内存）。

        边界（数据流）：tokens 先 reshape(-1) 压平，因此 (S,) 与 (1, S)
        两种输入形状都被接受；numpy 后端无需设备搬运，天然满足
        "CPU Offload 后仍在主机内存"的 §53 前置。
        """
        model.prefill_calls += 1
        kv = model.next_kv(np.asarray(tokens).reshape(-1)).transpose(1, 0, 2, 3)
        # einsum 输出 (S, L, H, D) → 转置为仓库约定 (L, S, H, D)
        return np.ascontiguousarray(kv)

    def decode(self, model: NumpyFakeModel, tokens, past_key_values) -> tuple[Any, Any]:
        """单步 decode：追加一行 KV，输出 (next_token, new_pkv)（§53）。

        缓存复用（关键路径）：
            1. 取当前 token（边界：接受 (1,)/(S,) 序列，取第 0 个）。
            2. 只对该 token 计算新一行 KV (L, H, D)。
            3. 沿序列维 axis=1 拼接到旧 PKV —— 这就是"缓存复用"：
               历史 token 的 KV 直接沿用，不重算。
            4. 用最新一行 KV 贪心 argmax 取 next_token（greedy decode）。
        §52 禁止 1：只消费 past_key_values 与当前 token —— 不读 teacher tokens。
        """
        model.decode_calls += 1
        tok = int(np.asarray(tokens).reshape(-1)[0])
        new_row = model.next_kv(np.array([tok]))[0]  # (L, H, D)
        # 序列维拼接：old (L, S, H, D) + new_row[:, None] (L, 1, H, D) → (L, S+1, H, D)
        pkv = np.concatenate(
            [np.asarray(past_key_values), new_row[:, None, :, :]], axis=1
        )
        token_out = int(np.argmax(model.logits(pkv)))
        return token_out, pkv

    def inject(self, model: NumpyFakeModel, kv) -> object:
        """§53 Inject：形状校验后把 (L_s, S, H, D) KV 挂到 Student 并返回。

        校验失败显式 raise，不静默降级（§52 禁止 8 的精神）。
        边界（inject 校验）：只校验 ndim == 4 且层数 L 匹配 Student 的
        n_layers；序列维 S 与 head 维 H 允许与映射器输出一致即可
        （S 由 Teacher 输入长度决定，H 由 Student 配置决定）。
        校验通过后挂到 model.injected_kv，供测试检查注入内容。
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
        """§53 Teacher Unload：numpy 无显存，仅清引用 + 归零计数器。

        计数器归零后，若同一模型实例被复用作 Student，其 prefill_calls
        会从 0 重新累计 —— 保证 §52 zero_prefill 断言不受 Teacher 阶段污染。
        """
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

    与 KV 缓存的交互（真实路径语义）：
        - KV 缓存形态为 HF 的 tuple of tensors / DynamicCache，
          shape 约定 (L, S, H, D) 与本仓库 numpy 契约一致。
        - forward_prefill 后 Teacher 的 KV 仍在显存，先 CPU offload
          （或直接读回主机内存）再 unload 模型。
        - inject 把映射后的 KV 写进 Student 的显存缓存对象，
          decode 阶段只逐 token 追加新行并复用旧行 —— 与 numpy
          后端行为同构，接入后即可端到端替换。

    骨架阶段一律显式 raise NotImplementedError，禁止静默 mock（§75 诚实性）。
    真实实现以 numpy 后端为行为参考，接入后即可端到端替换。
    """

    name: str = "torch"

    def load_model(self, run: dict[str, Any]) -> object:
        """§53 Teacher/Student Load：真实 HF 模型加载未实现，显式 raise。

        边界：torch 分支的 import 放函数内 try —— CPU/CI 环境没有 torch
        也能安全 import 本模块；这里仅探测 torch.cuda 可用性留作真实
        实现的入口，随后照常显式 raise（骨架不假装能跑 GPU）。
        """
        try:
            import torch  # type: ignore

            _ = torch.cuda.is_available()
        except ImportError:
            pass
        raise NotImplementedError(
            "torch 后端为接口骨架：真实 GPU 推理需要 transformers + CUDA（§53）。本骨架不实现无法验证的大型 forward；numpy 后端用于测试/CI。"
        )
