"""HF 模型加载 / KV 捕获共用工具（modelscope 源）。

═══════════════════════════════════════════════════════════════════════════════
被 `apcs/providers/hf_kv.py`（KVProvider）与 `apcs/inference/backends.py`
（TorchBackend）共用：

    - load_hf_model : 按 model_id 加载因果 LM（modelscope.cn 优先，HF 次之）
    - resolve_device: 从 cfg 解析 CUDA device，并校验 GPU 真正可计算（§75）
    - capture_kv_pair: 对单个模型 forward，抓 (L, S, H, 2D) 联合 KV numpy

KV 形状约定：每层 K 与 V 沿最后维拼接 → (L, S, H, 2*head_dim)，K 在前。
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _cuda_compute_available() -> str:
    """检测 CUDA 是否真正可用于计算；返回空串表示可用，否则返回原因。

    torch.cuda.is_available() 只报告"设备可见"，不保证能跑 kernel：
    编译时未包含该架构的 cubin/PTX（如 RTX 5090 sm_120 + torch 2.5.1）
    时任何实际算子都会报 `no kernel image is available`。提前给出可读
    失败原因（§75 诚实性：不静默假装 GPU 就绪）。
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
        supported = {a.replace("sm_", "") for a in archs}
        if cap_str not in supported:
            return (
                f"GPU 架构 sm_{cap_str} 与当前 torch({torch.__version__}) 编译支持 "
                f"({sorted(archs)}) 不匹配：kernel 无法加载，需升级 torch（≥2.7/cu128+）"
            )
    except Exception as e:  # noqa: BLE001
        return f"CUDA 能力检测失败：{type(e).__name__}: {e}"
    return ""


def resolve_device(cfg: dict[str, Any]) -> str:
    """从 cfg 解析 device，并校验 GPU 可计算（不可用则显式 raise）。

    device 默认 "cuda:0"（若 cfg.teacher.device_map 指定则用之）；
    校验失败抛 RuntimeError，附可读原因（§75 诚实性）。
    """
    device = str(
        cfg.get("teacher", {}).get("device_map", "cuda:0")
    ).split(":")[0] + ":0"
    reason = _cuda_compute_available()
    if reason:
        raise RuntimeError(
            f"provider 的 HF 路径无法使用 GPU：{reason}。"
            "请升级 PyTorch 至支持该 GPU 架构的版本（RTX 50 系需 torch≥2.7/cu128+），"
            "或显式改 cfg 走 synthetic 路径（provider.json 会记录该选择）。"
        )
    return device


def load_hf_model(
    model_id: str,
    revision: str,
    dtype,
    device: str,
    cls=None,
) -> Any:
    """按 model_id 加载因果 LM（modelscope.cn 优先，transformers/HF 次之）。

    cls: AutoModelForCausalLM 类（由调用方从 modelscope/transformers 选定，
    保持两源一致性）；为 None 时内部自动探测。
    """
    if cls is None:
        try:
            from modelscope import AutoModelForCausalLM as M  # type: ignore
        except ImportError:
            from transformers import AutoModelForCausalLM as M  # type: ignore
        cls = M
    model = cls.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=dtype,
        device_map={"": device},
    )
    model.eval()
    return model


def capture_kv_pair(model, input_ids) -> np.ndarray:
    """对单个模型 forward，抓 (L, S, H, 2D) 联合 KV numpy（宿主内存）。

    与 inference.backends._extract_kv_numpy 口径一致：每层 K|V 沿最后维
    拼接，转 (S, H, 2D) 后叠成 (L, S, H, 2D) float32。
    """
    import torch  # type: ignore

    with torch.no_grad():
        out = model(input_ids, use_cache=True)
    pkv = out.past_key_values
    n_layers = int(getattr(model.config, "num_hidden_layers", 0))
    if n_layers == 0:
        n_layers = len(pkv)
    return _extract_kv_numpy(pkv, n_layers)


def _extract_kv_numpy(pkv, n_layers: int) -> np.ndarray:
    """把 HF past_key_values（DynamicCache / tuple of (K,V)）转 (L, S, H, 2D)。"""
    import torch  # type: ignore

    layers = []
    if hasattr(pkv, "key_cache"):  # DynamicCache
        keys, vals = pkv.key_cache, pkv.value_cache
        for k, v in zip(keys, vals):
            k = k[0]
            v = v[0]
            kv_row = torch.cat([k, v], dim=-1)
            layers.append(kv_row.permute(1, 0, 2))
    else:  # tuple of (K, V) per layer
        for k, v in pkv:
            k = k[0]
            v = v[0]
            kv_row = torch.cat([k, v], dim=-1)
            layers.append(kv_row.permute(1, 0, 2))
    if len(layers) < n_layers:
        for _ in range(n_layers - len(layers)):
            layers.append(torch.zeros_like(layers[0]))
    # 显式转 float32 再 .numpy()：bf16 张量直接 .numpy() 会抛
    # "Got unsupported ScalarType BFloat16"（numpy 原生不支持 bfloat16）。
    stack = torch.stack(layers[:n_layers]).float().cpu().numpy()
    return np.ascontiguousarray(stack, dtype=np.float32)


__all__ = ["load_hf_model", "resolve_device", "capture_kv_pair", "_cuda_compute_available"]