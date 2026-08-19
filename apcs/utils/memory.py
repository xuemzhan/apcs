"""Memory-aware default calibration sizing。

═══════════════════════════════════════════════════════════════════════════════
设计目标：mapper runner 的默认 `n_calib`、`seq` 都对显存/RAM 极敏感，
但仓库没有 `pip install psutil` 这样的轻量探测。提供一个**纯标准库**的
轻量估算：当用户没在 cfg 里显式指定时，自动用一个保守的安全值。

**为什么不给一个全局大默认值**：
SDK / CI 容器常常 4 GB RAM 或更少。固定 128 样本 × 36 层 × 512 seq
× 8 H × 128 D × 4 B ≈ 7.5 GB —— 大多数环境会 OOM（实测：本次跑 t04 时
`MemoryError: Unable to allocate 4.00 MiB for an array with shape
(512, 8, 128) and data type float64`）。

**策略**：
- 检测进程可用内存（标准库 `resource` / 跨平台 fallback 到 1 GB 假设）；
- 按 KV 体积公式 `n_calib × seq × H × D × n_layers × 4 bytes` 算出
  安全上限；
- 用户在 cfg 显式设的值**优先**，本函数只在缺省时降级。
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os
import sys


def _available_memory_bytes() -> int:
    """返回**当前进程可用**内存估算（字节）。

    跨平台策略（按可靠性降序）：
        1. Linux   : /proc/meminfo + cgroup v2（如有）
        2. macOS   : sysctl 物理内存 × 1/2
        3. Windows : Get-ProcessMemoryInfo 全局可用内存
        4. fallback: 假设 1 GB（保守 CI 默认值）
    """
    try:
        if sys.platform == "linux":
            # /proc/meminfo 在所有现代 Linux 都有
            with open("/proc/meminfo", "rb") as f:
                for line in f:
                    if line.startswith(b"MemAvailable:"):
                        return int(line.split()[1]) * 1024  # kB → bytes
            # 容器内 /proc/meminfo 无 MemAvailable（老内核）→ 退回 MemFree
            with open("/proc/meminfo", "rb") as f:
                for line in f:
                    if line.startswith(b"MemFree:"):
                        return int(line.split()[1]) * 1024
        elif sys.platform == "darwin":
            import subprocess

            r = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True
            )
            if r.returncode == 0:
                return int(int(r.stdout.strip()) * 0.5)  # 保守：只许用 1/2
        elif sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", wintypes.DWORD),
                    ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_uint64),
                    ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64),
                    ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64),
                    ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return int(stat.ullAvailPhys)
    except Exception:  # noqa: BLE001
        pass
    # 最终 fallback：1 GB
    return 1 << 30


def safe_calibration_defaults(
    *,
    n_t: int,
    n_s: int,
    H: int,
    D: int,
    seq: int,
    budget_fraction: float = 0.10,
    n_calib_max: int = 128,
) -> dict[str, int]:
    """按可用内存返回保守的 n_calib / seq 建议值。

    输入：架构参数（n_t / n_s / H / D）+ 期望 seq
    输出：dict(可作为 cfg 的覆盖项)：
        - n_calib_estimate: 估计最大可容纳的校准样本数；
        - seq_estimate:     如 seq 太大则降级到安全值。

    内存预算只占可用内存的 `budget_fraction`（默认 10%，保守），
    留 90% 给 numpy 内部 / 矩阵求逆 / 进程栈 / 同时打开多个数组。

    公式（per-bug-3 聚合）：
        一次 fit_ridge_aggregate 需要把 (n_calib, seq, H, D) 张量同时驻留，
        加上等价的 Teacher 张量 + 临时 X^T X / X^T Y 矩阵 —— 经验上限 8× 系数。

    n_calib 上界 n_calib_max：设计规格 §32 要求 100–500 校准样本，
    默认 128（充裕内存直达上限），保证 8 ≤ n_calib ≤ 128 的安全区间。
    """
    avail = _available_memory_bytes()
    budget = int(avail * budget_fraction)
    # 估算当前 (n_calib, seq) 所需内存（factor 8 = 双倍 K/V + teacher + temps + 4× 内部）
    per_sample_full = seq * H * D * 4 * 8  # factor 8 for safety
    # 适配 (n_t, n_s)：逐层 fit 共 n_s 层（每层一份张量）
    size_all_layers = per_sample_full * max(n_t, n_s)
    if size_all_layers <= 0:
        return {"n_calib_estimate": 8, "seq_estimate": 128}
    n_calib = max(1, budget // size_all_layers)
    # 下限 8（保证 ridge 至少 8 个样本）；上限 n_calib_max（默认 128）
    n_calib = max(8, min(n_calib_max, n_calib))
    # 若 seq 太大、n_calib 跌出下限，缩 seq
    if seq > 1024 and n_calib < 16:
        safe_seq = 256
    elif seq > 2048:
        safe_seq = 512
    else:
        safe_seq = seq
    return {"n_calib_estimate": int(n_calib), "seq_estimate": int(safe_seq)}


def memory_status() -> dict[str, int]:
    """返回当前可用内存（MB）与合理预算。便于 CLI / log 报告。"""
    avail = _available_memory_bytes()
    return {
        "available_mb": int(avail / (1024 * 1024)),
        "budget_mb": int(avail * 0.25 / (1024 * 1024)),
    }


__all__ = ["safe_calibration_defaults", "memory_status", "_available_memory_bytes"]
