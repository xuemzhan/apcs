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


class HandoffPipeline:
    """单卡 handoff 推理管线（design.md §53 顺序固定）。

    参数：
        backend: 推理后端（InferenceBackend；numpy 可跑 / torch 骨架）
        mapper:  具备 transform(kv_t, layer_map) 的映射器（如 RidgeMapper）
        layer_map: Student 层 → Teacher 层索引（§21 proportional_mapping 产物）
        cfg:     模型配置，须含 teacher / student 两节的 num_layers 等
        out_dir: 产物目录（metrics.json + summary.md，§63 风格）
    """

    def __init__(
        self, backend, mapper, layer_map, cfg: dict[str, Any], out_dir
    ) -> None:
        self.backend = backend
        self.mapper = mapper
        self.layer_map = layer_map
        self.cfg = cfg
        self.out_dir = out_dir

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _model_run(self, role: str) -> dict[str, Any]:
        """构造单侧模型参数（§53 Load 的 run 参数），并注入 role/seed。"""
        run = dict(self.cfg.get(role, {}))
        run["role"] = role
        run.setdefault("seed", 1 if role == "teacher" else 2)
        return run

    @contextlib.contextmanager
    def _stage(self, name: str, timings: dict[str, float]):
        """§49 计时块：进入/退出都 sync，记录 elapsed ms。"""
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
        """
        timings: dict[str, float] = {}

        # ① Teacher Load → Forward → Capture（§53 前四步）
        with self._stage("teacher_load", timings):
            teacher = self.backend.load_model(self._model_run("teacher"))
        with self._stage("forward_capture", timings):
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
            kv_s = self.mapper.transform(kv_t, self.layer_map)
        with self._stage("inject", timings):
            injected = self.backend.inject(student, kv_s)

        # ③ Decode：只消费注入 KV，绝不 re-prefill（§52 禁止 1 / zero prefill）
        generated: list[int] = []
        with self._stage("decode", timings):
            pkv = injected
            tok = student_prefix
            for _ in range(int(n_gen)):
                tok, pkv = self.backend.decode(student, tok, pkv)
                generated.append(int(np.asarray(tok).reshape(-1)[0]))

        # ④ metrics（§52 统计断言 + §49 计时 + §75 offline_demo 标注）
        teacher_ids = np.asarray(teacher_tokens).reshape(-1)
        metrics: dict[str, Any] = {
            "task": "inference",
            "backend": self.backend.name,
            "offline_demo": self.backend.name == "numpy",
            "note": (
                "numpy 后端为小型假模型，仅工程骨架演示，非真实 GPU 证据（§75）；"
                "torch 后端为接口骨架，真实 GPU 推理待接线（§53）"
            ),
            "n_teacher_tokens": int(teacher_ids.size),
            "n_student_prefix": int(np.asarray(student_prefix).reshape(-1).size),
            "n_generated": len(generated),
            "n_teacher_layers": int(np.asarray(kv_t).shape[0]),
            "n_student_layers": int(np.asarray(kv_s).shape[0]),
            "kv_t_shape": list(np.asarray(kv_t).shape),
            "kv_s_shape": list(np.asarray(kv_s).shape),
            "injected_kv_shape": list(np.asarray(injected).shape),
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
        """§63 风格的人类可读摘要。"""
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
