"""§53 单卡 handoff 推理管线（apcs/inference/pipeline.py）。

═══════════════════════════════════════════════════════════════════════════════
本模块对应 design.md：
    §53 单卡执行策略（强制顺序）：
        Teacher Load → Forward → Capture → CPU Offload → Teacher Unload →
        CUDA Cleanup → Student Load → Map → Inject → Decode
    §52 禁止 1 —— Student 在 decode 阶段不重新读 X（zero prefill）：
        管线只把注入的 KV 状态传给 backend.decode，Student 侧
        从不调用 forward_prefill；结果以计数器做统计断言。
    §49 计时规范 —— 每个阶段边界调用 backend.sync() 后计时。
    §75 诚实性 —— numpy 后端为 offline_demo；torch 后端骨架显式 raise。

用法：传入真实 backend / mapper / layer_map / cfg 即可端到端跑（§53 骨架）。
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import contextlib
import time
from typing import Any

import numpy as np

from ..io.runs import write_json, write_text

_ORDER = (
    "teacher_load",
    "forward_capture",
    "teacher_unload",
    "cuda_cleanup",
    "student_load",
    "map",
    "inject",
    "decode",
)
# 上表为 §53 强制阶段顺序（8 阶段）：Teacher Load → Forward → Capture →
# CPU Offload → Teacher Unload → CUDA Cleanup → Student Load → Map →
# Inject → Decode。管线按此顺序执行，不提供跳过/乱序的入口。


class HandoffPipeline:
    """单卡 handoff 推理管线（design.md §53 顺序固定）。

    参数：
        backend: 推理后端（InferenceBackend；numpy 可跑 / torch 骨架）
        mapper:  具备 transform(kv_t, layer_map) 的映射器。
                 ◆ D2 修复说明：TorchBackend.forward_prefill 产出的是
                 (L_t, S, H, 2D) 的 K‖V 联合布局，必须搭配
                 apcs.mapper.joint.JointKVMapper（内部拆 K/V 独立变换，
                 §22）—— 直接传 per-kind mapper 会在 einsum 处形状崩溃。
                 NumpyBackend 产出 (L_t, S, H, D) 假 KV，用普通 mapper。
        layer_map: Student 层 → Teacher 层索引（§21 proportional_mapping 产物）
        cfg:     模型配置，须含 teacher / student 两节的 num_layers 等
        out_dir: 产物目录（metrics.json + summary.md，§63 风格）

    KV 缓存数据流（管线视角）：
        Teacher Prefill → kv_t (L_t, S, H, 2D)  ← Teacher 自产 KV（Capture，K‖V 拼接）
        mapper.transform → kv_s (L_s, S, H, 2D) ← 跨模型映射（JointKVMapper 拆 K/V）
        backend.inject  → Student 侧可消费 PKV ← 缓存复用入口（注入）
        backend.decode  → 逐 token 拼接新行     ← 只复用注入 KV，不 re-prefill
    """

    def __init__(
        self, backend, mapper, layer_map, cfg: dict[str, Any], out_dir
    ) -> None:
        # 记录依赖并原样转交 run()：backend 负责模型/KV 进出，
        # mapper + layer_map 负责 Teacher KV → Student KV 的映射
        self.backend = backend
        self.mapper = mapper
        self.layer_map = layer_map
        self.cfg = cfg
        self.out_dir = out_dir

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _model_run(self, role: str) -> dict[str, Any]:
        """构造单侧模型参数（§53 Load 的 run 参数），并注入 role/seed。

        边界（run 参数检查）：cfg 缺某侧配置节时取空 dict，随后由
        后端 load_model 负责校验必需键 —— 管线不替后端做默认值兜底。
        seed 默认值按角色区分（teacher=1 / student=2），保证 Teacher 与
        Student 的假模型权重不同但各自可复现。
        """
        run = dict(self.cfg.get(role, {}))
        run["role"] = role
        run.setdefault("seed", 1 if role == "teacher" else 2)
        return run

    @contextlib.contextmanager
    def _stage(self, name: str, timings: dict[str, float]):
        """§49 计时块：进入/退出都 sync，记录 elapsed ms。

        计时边界：进入时先 sync 把前一阶段异步工作排干，退出时再 sync
        保证本阶段 GPU kernel 全部完成 —— 两个 sync 之间的墙钟即为
        该阶段耗时；异常时 finally 仍会收尾并记录，异常本身照常上抛。
        """
        self.backend.sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.backend.sync()
            timings[name] = (time.perf_counter() - t0) * 1000.0

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def run(self, teacher_tokens, student_prefix, n_gen: int) -> dict[str, Any]:
        """按 §53 顺序执行完整 handoff，返回 metrics。

        参数：
            teacher_tokens: Teacher 输入 token id 序列（(S,) 或 (1, S)）
            student_prefix: decode 起始 token（如 teacher 最后一 token）
            n_gen:          生成步数
        返回：
            metrics dict：n_generated / n_teacher_tokens / 注入 KV 形状 /
            layer_map / §52 zero_prefill gate / 阶段计时（§49）

        边界（数据流）：
            - kv_t 形状 (L_t, S, H, D)；mapper.transform 按 layer_map 做
              层间映射/聚合（拼接/池化见 mapper 实现），序列维 S 不变。
            - decode 只消费注入的 PKV：每步返回新 PKV 作为下一步输入，
              Student 全程不调用 forward_prefill（zero prefill，§52 禁止 1）。
            - n_gen < 1 时 decode 阶段为空，但 Teacher 捕获与注入照常完成，
              此时 generated == []、student_decode_calls == 0。
        """
        timings: dict[str, float] = {}

        # ① Teacher Load → Forward → Capture（§53 前四步）
        with self._stage("teacher_load", timings):
            teacher = self.backend.load_model(self._model_run("teacher"))
        with self._stage("forward_capture", timings):
            # Teacher 侧 Prefill，产出 KV 缓存 kv_t：(L_t, S, H, D)
            kv_t = self.backend.forward_prefill(teacher, teacher_tokens)
        with self._stage("teacher_unload", timings):
            # §53 CPU Offload：capture 产物（numpy ndarray）天然在主机内存
            self.backend.unload(teacher)
        with self._stage("cuda_cleanup", timings):
            # §53 CUDA Cleanup + §49 sync（numpy no-op，torch 需 empty_cache）
            self.backend.sync()

        # ② Student Load → Map → Inject（§53 后三步前半）
        with self._stage("student_load", timings):
            student = self.backend.load_model(self._model_run("student"))
        with self._stage("map", timings):
            # 跨模型映射：kv_t (L_t, S, H, D) → kv_s (L_s, S, H, D)，
            # 仅做层间线性组合/池化，序列维 S 与 head 结构按 layer_map 对齐
            kv_s = self.mapper.transform(kv_t, self.layer_map)
        with self._stage("inject", timings):
            # 缓存复用入口：注入成功才允许后续 decode；
            # 形状不匹配时后端显式 raise，绝不静默 re-prefill（§52 禁止 8）
            injected = self.backend.inject(student, kv_s)

        # ③ Decode：只消费注入 KV，绝不 re-prefill（§52 禁止 1 / zero prefill）
        generated: list[int] = []
        with self._stage("decode", timings):
            pkv = injected
            tok = student_prefix
            for _ in range(int(n_gen)):
                # 每步复用当前 PKV（含全部历史行）并追加当前 token 的新行；
                # 返回的 new_pkv 作为下一步缓存继续复用（序列维逐 token 增长）
                tok, pkv = self.backend.decode(student, tok, pkv)
                generated.append(int(np.asarray(tok).reshape(-1)[0]))

        # ④ metrics（§52 统计断言 + §49 计时 + §75 offline_demo 标注）
        teacher_ids = np.asarray(teacher_tokens).reshape(-1)
        # 注入产物可能是 numpy ndarray（numpy 后端）或 HF DynamicCache（torch 后端），
        # 形状上报统一转 numpy（torch 后端已把 KV 以 numpy 挂到 model.injected_kv）
        injected_arr = getattr(student, "injected_kv", None)
        if injected_arr is not None:
            injected_shape = list(np.asarray(injected_arr).shape)
        else:
            try:
                injected_shape = list(np.asarray(injected).shape)
            except Exception:  # noqa: BLE001  DynamicCache 等不可直接 asarray
                injected_shape = [int(s) for s in np.asarray(kv_s).shape]
        metrics: dict[str, Any] = {
            "task": "inference",
            "backend": self.backend.name,
            "offline_demo": self.backend.name == "numpy",
            "note": (
                "numpy 后端为小型假模型，仅工程骨架演示，非真实 GPU 证据（§75）；"
                "torch 后端为真实 HF 模型推理路径（modelscope 源），可作 GPU 证据"
            ),
            "n_teacher_tokens": int(teacher_ids.size),
            "n_student_prefix": int(np.asarray(student_prefix).reshape(-1).size),
            "n_generated": len(generated),
            "n_teacher_layers": int(np.asarray(kv_t).shape[0]),
            "n_student_layers": int(np.asarray(kv_s).shape[0]),
            "kv_t_shape": list(np.asarray(kv_t).shape),
            "kv_s_shape": list(np.asarray(kv_s).shape),
            "injected_kv_shape": injected_shape,
            "layer_map": self.layer_map,
            "tokens": generated,
            "gates": {"zero_prefill": student.prefill_calls == 0},
            "student_prefill_calls": int(student.prefill_calls),
            "student_decode_calls": int(student.decode_calls),
            "timings_ms": timings,
        }
        if self.out_dir is not None:
            write_json(self.out_dir / "metrics.json", metrics)
            write_text(self.out_dir / "summary.md", self._summary(metrics))
        return metrics

    @staticmethod
    def _summary(m: dict[str, Any]) -> str:
        """§63 风格的人类可读摘要。

        从 metrics dict 提取关键字段：backend / 注入 KV 形状 / layer_map
        长度 / zero_prefill gate / 阶段计时 —— 全部来自 run() 已算好的
        指标，不重复计算（纯展示函数）。
        """
        return (
            "# §53 单卡推理管线骨架\n\n"
            f"- backend: {m['backend']} (offline_demo={m['offline_demo']})\n"
            f"- teacher tokens: {m['n_teacher_tokens']}, "
            f"generated: {m['n_generated']}\n"
            f"- 注入 KV 形状: {m['injected_kv_shape']} "
            f"(student 层 {m['n_student_layers']}, layer_map 长度 {len(m['layer_map'])})\n"
            f"- §52 禁止 1 zero_prefill gate: {m['gates']['zero_prefill']}\n"
            f"- 阶段计时 (ms, §49): {m['timings_ms']}\n"
        )
