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
# torch 后端（真实 GPU 路径，§53）
# ---------------------------------------------------------------------------


def _cuda_compute_available() -> str:
    """检测 CUDA 是否真正可用于计算；返回空串表示可用，否则返回原因。

    torch.cuda.is_available() 只报告"设备可见"，不保证能跑 kernel：
    编译时未包含该架构的 cubin/PTX（如 RTX 5090 sm_120 + torch 2.5.1）
    时，任何实际算子都会报 `no kernel image is available`。这里用
    `torch.cuda.get_device_capability` 与 `get_arch_list` 交叉核对，
    提前给出可读的失败原因（§75 诚实性：不静默假装 GPU 就绪）。
    """
    try:
        import torch  # type: ignore
    except ImportError:
        return "未安装 torch"
    if not torch.cuda.is_available():
        return "CUDA 不可用（torch.cuda.is_available()==False）"
    try:
        cap = torch.cuda.get_device_capability(0)
        # sm 后缀无小数点：cap (12,0) → "120"（对应 arch "sm_120"），
        # cap (9,0) → "90"。旧写法 f"{cap[0]}.{cap[1]}" → "12.0" 永远匹配不上。
        cap_str = f"{cap[0]}{cap[1]}"
        archs = torch.cuda.get_arch_list()
        # get_arch_list 形如 ['sm_80','sm_90']；Blackwell 需 sm_120。
        supported = {a.replace("sm_", "") for a in archs}
        if cap_str not in supported:
            # sm_89 (Ada Lovelace/RTX 4090) 兼容 sm_86/sm_90 kernel，
            # 实测可运行，跳过严格检查。
            major = cap[0]
            compatible = any(
                s.startswith(str(major)) for s in supported
            )
            if not compatible:
                return (
                    f"GPU 架构 sm_{cap_str} 与当前 torch({torch.__version__}) 编译支持 "
                    f"({sorted(archs)}) 不匹配：kernel 无法加载，需升级 torch（≥2.7/cu128+）"
                )
    except Exception as e:  # noqa: BLE001
        return f"CUDA 能力检测失败：{type(e).__name__}: {e}"
    return ""


class TorchModel:
    """torch 后端的模型包装：HF 模型 + tokenizer + §52 可观测计数器。

    为什么要包装而非直接返回 HF model：
        - 管线（HandoffPipeline）直接访问 `student.prefill_calls` /
          `student.decode_calls` 做 §52 zero-prefill 统计断言；HF 模型
          没有这些属性，包装对象补上（NumpyFakeModel 同款契约）。
        - 保存 tokenizer / device / dtype 供 forward/decode/inject 复用，
          避免每次重新解析 cfg。
    """

    def __init__(self, model, tokenizer, *, role: str, device: str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.role = role
        self.device = device
        # §52 可观测计数器（与 NumpyFakeModel 对齐）
        self.prefill_calls = 0
        self.decode_calls = 0
        # inject 后挂载的 KV（numpy 宿主内存，供测试检查；真实 decode 用它建缓存）
        self.injected_kv: np.ndarray | None = None


def _extract_kv_numpy(pkv, n_layers: int, dtype_bytes: int = 2) -> np.ndarray:
    """把 HF past_key_values（DynamicCache / tuple of (K,V)）转成 (L, S, H, D) numpy。

    输入 pkv 的元素是 (batch, num_kv_heads, seq_len, head_dim) 的张量；
    每个 layer 取 K 与 V，沿最后维拼接成 (S, H, 2*head_dim) 的"联合 KV 行"，
    再叠成 (L, S, H, 2*head_dim)。返回 float32，值为 K 在前 V 在后。

    为什么不拆开 K/V 各存一份 (L,S,H,D)：
        仓库 numpy 契约以 `(L, S, H, D)` 表达每层一份 KV 状态（synthetic
        与 mapper 均如此），真实 K/V 是"每层两份"。合并成 K|V 联合行可
        保持同一契约（D'=2*head_dim），映射/注入两端口径一致。
    """
    import torch  # type: ignore

    layers = []
    if hasattr(pkv, "key_cache"):  # DynamicCache
        keys, vals = pkv.key_cache, pkv.value_cache
        for k, v in zip(keys, vals):
            k = k[0]  # drop batch dim → (H, S, D)
            v = v[0]
            kv_row = torch.cat([k, v], dim=-1)  # (H, S, 2D)
            layers.append(kv_row.permute(1, 0, 2))  # (S, H, 2D)
    else:  # tuple of (K, V) per layer
        for k, v in pkv:
            k = k[0]
            v = v[0]
            kv_row = torch.cat([k, v], dim=-1)
            layers.append(kv_row.permute(1, 0, 2))
    if len(layers) < n_layers:
        # 若模型只缓存了部分层（骨架用法），缺失层补零行保持形状契约
        for _ in range(n_layers - len(layers)):
            layers.append(torch.zeros_like(layers[0]))
    stack = torch.stack(layers[:n_layers]).float().cpu().numpy()
    return np.ascontiguousarray(stack, dtype=np.float32)


def _build_cache_from_kv(
    kv: np.ndarray, device: str | None = None, dtype: Any | None = None
) -> Any:
    """把 numpy (L, S, H, 2D) 联合 KV 行还原成 HF DynamicCache。

    与 _extract_kv_numpy 互为逆操作：沿最后维切成 K|V，各转置成
    (1, H, S, D)，组装成 transformers.cache_utils.DynamicCache 供 decode。
    无 transformers 时退回轻量列表结构（接口等价，`tuple of (K,V)`）。
    device 非空时把缓存张量搬到目标设备（否则与 GPU 上模型的 decode
    拼接缓存时触发 "Expected all tensors to be on the same device"）；
    dtype 指定时按模型精度铸造（bf16 模型 + float32 缓存会让 SDPA 抛
    "Expected query, key, and value to have the same dtype"）。
    """
    import torch  # type: ignore

    kv = np.asarray(kv)  # (L, S, H, 2D)
    L, S, H, D2 = kv.shape
    D = D2 // 2
    try:
        from transformers.cache_utils import DynamicCache  # type: ignore
    except ImportError:
        DynamicCache = None  # type: ignore[assignment]

    kt = torch.from_numpy(kv[:, :, :, :D]).permute(0, 2, 1, 3)  # (L,H,S,D)
    vt = torch.from_numpy(kv[:, :, :, D:]).permute(0, 2, 1, 3)
    if dtype is not None:
        kt = kt.to(dtype)
        vt = vt.to(dtype)
    else:
        kt = kt.to(torch.float32)
        vt = vt.to(torch.float32)
    if device is not None:
        kt = kt.to(device)
        vt = vt.to(device)
    keys = [k.unsqueeze(0) for k in kt]  # (1,H,S,D) per layer
    vals = [v.unsqueeze(0) for v in vt]
    if DynamicCache is not None:
        cache = DynamicCache()
        cache.update(keys[0], vals[0], 0)
        for layer_idx in range(1, L):
            cache.update(keys[layer_idx], vals[layer_idx], layer_idx)
        return cache
    return list(zip(keys, vals))  # type: ignore[return-value]


class TorchBackend(InferenceBackend):
    """torch 后端：真实 HF 模型 + CUDA 的 §53 推理路径。

    以 numpy 后端为行为参考，但模型从 modelscope/transformers 加载：
        - load_model      → AutoModelForCausalLM.from_pretrained（modelscope 优先）
        - forward_prefill → 真实 attention，抓 past_key_values 转 numpy 宿主内存
        - inject          → 把映射后 (L_s,S,H,2D) 联合 KV 还原成 DynamicCache
        - decode          → 逐 token 增量生成，复用注入 KV（zero prefill）
        - unload          → 释放模型 + torch.cuda.empty_cache()
        - sync            → torch.cuda.synchronize()（§49）

    与 KV 缓存的交互（真实路径语义）：
        - KV 以 numpy (L, S, H, 2*head_dim) 在主机内存传递（capture/offload），
          显存中仅在 decode 阶段以 DynamicCache 短期存在。
        - forward_prefill 后 Teacher 的 KV 读回主机，再 unload 释放显存，
          避免 Teacher/Student 同时在显存造成峰值超限（§53）。

    §75 诚实性：
        - GPU 不可计算（sm_120 与 torch 版本不匹配）时 load_model 显式 raise，
          附可读原因，禁止静默假装 GPU 就绪；
        - 模型参数缺失显式 raise，不得给默认 mock。
    """

    name: str = "torch"

    def __init__(self, device: str | None = None) -> None:
        super().__init__()
        # device 默认取 cuda:0；由 _cuda_compute_available 决定是否真的可用
        self._device = device or "cuda:0"

    def _resolve_device(self) -> str:
        """校验 CUDA 可用性并返回实际 device。

        校验失败抛 NotImplementedError（带可读原因），作为 load_model 的
        前置守卫 —— 让「GPU 路径不可用」在 load 阶段就显式失败（§75）。
        """
        reason = _cuda_compute_available()
        if reason:
            raise NotImplementedError(
                f"torch 后端无法使用 GPU：{reason}。"
                "请升级 PyTorch 至支持该 GPU 架构的版本（RTX 50 系需 torch≥2.7/cu128+）。"
            )
        return self._device

    def load_model(self, run: dict[str, Any]) -> TorchModel:
        """§53 Teacher/Student Load：加载真实 HF 因果 LM（modelscope 优先）。

        run 参数：model_id（必需）、revision/dtype/device_map（可选）、
        num_layers/num_kv_heads/head_dim/vocab_size（显式架构字段，可选）。
        模型源优先 modelscope.cn，其次 transformers（见 compat/scanner 同款策略）。
        """
        missing = [k for k in ("model_id",) if run.get(k) is None]
        if missing:
            raise NotImplementedError(
                f"torch 后端缺少模型参数 {missing}（role={run.get('role')}）："
                "必须显式传 model_id 才能加载真实模型（§75）"
            )
        device = self._resolve_device()
        import torch  # type: ignore

        model_id = str(run["model_id"])
        revision = str(run.get("revision", "main"))
        dtype_name = str(run.get("dtype", "bfloat16"))
        dtype = getattr(torch, dtype_name, torch.bfloat16)

        # 模型源：modelscope.cn 优先（AutoModelForCausalLM 与 transformers 接口对齐）
        try:
            from modelscope import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except ImportError:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=dtype,
            device_map={"": device},
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        model.eval()
        return TorchModel(model, tokenizer, role=str(run.get("role", "")), device=device)

    def forward_prefill(self, model: TorchModel, tokens) -> np.ndarray:
        """§53 Forward + Capture：真实 prefill，返回 (L_t, S, H, 2D) numpy KV。

        KV 已在宿主内存（读回 CPU），满足 §53 CPU Offload 前置。
        """
        kv, _, _ = self.forward_prefill_native(model, tokens)
        return kv

    def forward_prefill_native(self, model: TorchModel, tokens):
        """返回 KV numpy、模型原生 cache 与末位 logits，供 Gate 0 真对照。"""
        import torch  # type: ignore

        model.prefill_calls += 1
        ids = torch.as_tensor(np.asarray(tokens).reshape(1, -1), dtype=torch.long).to(
            model.device
        )
        with torch.no_grad():
            out = model.model(ids, use_cache=True)
        n_layers = int(getattr(model.model.config, "num_hidden_layers", 0))
        if n_layers == 0:
            n_layers = len(out.past_key_values)
        kv = _extract_kv_numpy(out.past_key_values, n_layers)
        logits = out.logits[:, -1, :].float().cpu().numpy()
        return kv, out.past_key_values, logits

    def decode(
        self, model: TorchModel, tokens, past_key_values
    ) -> tuple[Any, Any]:
        """单步 decode：复用注入 KV + 当前 token，追加一行，返回 (next_token, new_pkv)。

        §52 禁止 1：只消费 past_key_values 与当前 token，绝不重新读 X。
        """
        next_tok, cache, _ = self.decode_with_logits(model, tokens, past_key_values)
        return next_tok, cache

    def decode_with_logits(self, model: TorchModel, tokens, past_key_values):
        """单步 decode，同时返回 float32 logits；Gate 0 不再用 token 代替 logits。"""
        import torch  # type: ignore

        model.decode_calls += 1
        tok = int(np.asarray(tokens).reshape(-1)[0])
        cache = past_key_values
        if not isinstance(past_key_values, tuple) and not hasattr(
            past_key_values, "key_cache"
        ):
            cache = _build_cache_from_kv(
                past_key_values, device=model.device, dtype=model.model.dtype
            )
        ids = torch.as_tensor([[tok]], dtype=torch.long).to(model.device)
        with torch.no_grad():
            out = model.model(ids, past_key_values=cache, use_cache=True)
        logits_t = out.logits[:, -1].float()
        next_tok = int(torch.argmax(logits_t, dim=-1).item())
        return next_tok, out.past_key_values, logits_t.cpu().numpy()

    def inject(self, model: TorchModel, kv) -> object:
        """§53 Inject：把映射后 (L_s, S, H, 2D) 联合 KV 还原为可消费缓存。

        校验失败显式 raise（§52 禁止 8，不静默 re-prefill）。
        返回 DynamicCache（decode 复用）。
        """
        kv = np.asarray(kv)
        if kv.ndim != 4:
            raise NotImplementedError(
                f"注入 KV 形状 {kv.shape} 非法（需 (L, S, H, 2D)）：inject 必须显式校验，禁止静默 re-prefill（§52 禁止 8）"
            )
        cache = _build_cache_from_kv(kv, device=model.device, dtype=model.model.dtype)
        model.injected_kv = kv
        return cache

    def unload(self, model: TorchModel) -> None:
        """§53 Teacher Unload：释放模型 + empty_cache + 归零计数器。"""
        import torch  # type: ignore

        del model.model
        del model.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model.prefill_calls = 0
        model.decode_calls = 0

    def sync(self) -> None:
        """§49 计时边界同步：GPU 后端 torch.cuda.synchronize()。"""
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except ImportError:
            pass
