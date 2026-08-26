"""apcs.inference：§53 单卡推理管线（真实 GPU 路径的可测试接口骨架）。

═══════════════════════════════════════════════════════════════════════════════
对应 design.md：
    §53 单卡执行策略 —— Teacher Load → Forward → Capture → CPU Offload →
        Teacher Unload → CUDA Cleanup → Student Load → Map → Inject → Decode
    §52 禁止 1 —— Student 不重新读 X（zero prefill，decode 统计断言）
    §49 计时规范 —— backend.sync() 在阶段边界调用
    §75 诚实性 —— 骨架显式 raise，numpy 后端 offline_demo 标注

接线方式（design-gap-review.md P1）：
    传入真实 backend（numpy 可跑 / torch 待接线）、mapper（如 RidgeMapper）、
    layer_map（§21 proportional_mapping）、cfg 即可端到端跑：
        from apcs.inference import HandoffPipeline, NumpyBackend
        pipe = HandoffPipeline(NumpyBackend(), mapper, layer_map, cfg, out_dir)
        metrics = pipe.run(teacher_tokens, student_prefix, n_gen=64)
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from .backends import (
    InferenceBackend,
    NumpyBackend,
    NumpyFakeModel,
    TorchBackend,
    ZeroPrefillCounter,
    pkv_to_numpy,
)
from .pipeline import HandoffPipeline

__all__ = [
    "InferenceBackend",
    "NumpyBackend",
    "NumpyFakeModel",
    "TorchBackend",
    "HandoffPipeline",
    "ZeroPrefillCounter",
    "pkv_to_numpy",
]
