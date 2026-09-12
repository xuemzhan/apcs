#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# APCS 真实 GPU 一键实验入口（Linux GPU 机器用）
#
# 用法（在 GPU 机器仓库根目录）：
#   bash scripts/run_real_gpu.sh              # 标准档：calib=256 / eval=64 / ctx≈1024
#   APCS_CALIB=256 APCS_EVAL=64 bash scripts/run_real_gpu.sh
#   APCS_CTX4K=1 bash scripts/run_real_gpu.sh # 增加 4096 档（显存/时间 ×~4）
#   APCS_SKIP_PREPARE=1 ...                   # 跳过 prepare-data（已冻结清单时）
#
# 流程（§71 DAG + 注入评测）：
#   环境检查 → [prepare-data] → t00..t03 → inject-eval(真实注入评分, 导出审计 artifact)
#   → 派生运行时配置(挂 artifact 路径/score=artifact/demo 放行) → t04..t13
#
# 证据规则：
#   - 所有 handoff 评分来自真实 cache injection，zero-prefill 由计数器断言；
#   - t08 以 allow_synthetic_demo=true 放行（offline demo 标注，不作为证据）；
#   - t10 timing=hf 仍为 cuda_proxy 标注；t11 若缺端到端计时将输出 INCONCLUSIVE——
#     这是预期行为（诚实门控），不是脚本失败。
# ═══════════════════════════════════════════════════════════════════════════════
set -uo pipefail

BASE_CFG="configs/pair_qwen3_real_gpu.yaml"
RUN_CFG="configs/pair_qwen3_real_gpu_run.yaml"
CALIB="${APCS_CALIB:-256}"
EVAL_N="${APCS_EVAL:-64}"
INJECT_N="${APCS_INJECT_SAMPLES:-32}"

log()  { printf '\n\033[1;36m[apcs-run]\033[0m %s\n' "$*"; }
fail() { printf '\n\033[1;31m[apcs-run][ABORT]\033[0m %s\n' "$*"; exit 1; }

# ── 0. 环境检查 ────────────────────────────────────────────────────────────────
log "环境检查"
python - <<'PY' || fail "GPU 环境不满足：请先 pip install -e '.[gpu]' 并确认 CUDA 可用"
import torch
assert torch.cuda.is_available(), "CUDA 不可用"
p = torch.cuda.get_device_properties(0)
print(f"  GPU: {p.name}  VRAM: {p.total_memory/2**30:.1f} GB  sm_{p.major}{p.minor}")
import transformers, datasets  # noqa: F401
print(f"  transformers={transformers.__version__} datasets={datasets.__version__}")
PY

# ── 1. 数据准备（幂等） ────────────────────────────────────────────────────────
if [ "${APCS_SKIP_PREPARE:-0}" != "1" ]; then
  log "prepare-data（冻结 split manifest）"
  python -m apcs.cli prepare-data --config "$BASE_CFG" --new-run || fail "prepare-data 失败"
fi

# ── 2. 前置 DAG：t00 → t03（t00 开新 run，其后任务复用 sticky 指针） ───────────
log "运行 t00（建立新 run_id）"
python -m apcs.cli t00 --config "$BASE_CFG" --new-run || fail "t00 退出码=$?"
for t in t01 t02 t03; do
  log "运行 $t"
  python -m apcs.cli "$t" --config "$BASE_CFG"
  rc=$?
  [ "$rc" -eq 0 ] || { [ "$t" = "t02" ] && [ "$rc" -eq 1 ]; } || fail "$t 退出码=$rc（Gate 未过或被阻断）"
done
# t02 允许 rc=1 的说明：cosine>0.9999 为报告性阈值；其余任务失败即中止。

# ── 3. 解析 sticky run_id（t00 已建立指针） ───────────────────────────────────
BASE_DIR=$(python -c "import yaml;print(yaml.safe_load(open('$BASE_CFG'))['output']['base_dir'])")
EXP_NAME=$(python -c "import yaml;print(yaml.safe_load(open('$BASE_CFG'))['experiment']['name'])")
RUN_ID=$(cat "$BASE_DIR/$EXP_NAME.current" 2>/dev/null) || fail "未找到 run 指针 $BASE_DIR/$EXP_NAME.current"
RUN_DIR="$BASE_DIR/$RUN_ID"
log "run_id = $RUN_ID"

# ── 4. inject-eval：真实注入评测（zero-prefill 断言 + 审计 artifact 导出） ────
log "inject-eval（n=$INJECT_N，方法: student/teacher/text/ridge）"
APCS_INJECT_SAMPLES="$INJECT_N" python -m apcs.cli inject-eval --config "$BASE_CFG" \
  || fail "inject-eval 失败（检查模型缓存/GPU 显存）"

REPL_ART="$RUN_DIR/inject_eval/replacement_score_artifact.json"
CAP_ART="$RUN_DIR/inject_eval/capability_score_artifact.json"
[ -f "$REPL_ART" ] || fail "缺少 $REPL_ART"
[ -f "$CAP_ART" ]  || fail "缺少 $CAP_ART"
log "审计 artifact 就绪：replacement + capability"

# ── 5. 派生运行时配置（深合并 override 到 base） ──────────────────────────────
log "生成运行时配置 $RUN_CFG"
REPL_ART="$REPL_ART" CAP_ART="$CAP_ART" CALIB="$CALIB" EVAL_N="$EVAL_N" \
CTX4K="${APCS_CTX4K:-0}" python - <<'PY'
import os, yaml, copy
base = yaml.safe_load(open("configs/pair_qwen3_real_gpu.yaml"))
ov = {
    "datasets": {"fidelity": ["needle_longctx"]},
    "context_lengths": [512, 1024, 4096] if os.environ["CTX4K"] == "1" else [512, 1024],
    "gates": {"retention_min": 0.80, "retention_strong": 0.90, "chg_positive": True},
    "mapper": {
        "real_calibration_samples": int(os.environ["CALIB"]),
        "real_eval_samples": int(os.environ["EVAL_N"]),
    },
    "advantage": {"allow_synthetic_demo": True},
    "provider": {
        "kv": "hf", "score": "artifact", "timing": "hf",
        "replacement_score_artifact_path": os.environ["REPL_ART"],
        "score_artifact_path": os.environ["CAP_ART"],
    },
}
def merge(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            merge(dst[k], v)
        else:
            dst[k] = v
merge(base, ov)
base.setdefault("experiment", {})["description"] = (
    "One-click real-GPU run: functional compatibility + audited injection scoring"
)
yaml.safe_dump(base, open("configs/pair_qwen3_real_gpu_run.yaml", "w"), sort_keys=False,
               allow_unicode=True)
print("  written:", os.path.abspath("configs/pair_qwen3_real_gpu_run.yaml"))
PY

# ── 6. 主 DAG：t04..t13（同一 sticky run） ────────────────────────────────────
# 允许 rc=1（Gate CONDITIONAL/FAIL 但已产出诚实结论）的任务白名单：
#   t05(Gate1 可能 CONDITIONAL), t08(demo), t10(proxy), t11(INCONCLUSIVE 预期),
#   t12(placeholder→真实 hidden states 视导出而定), t13(<2 pair 告警)
ALLOW_RC1="t05 t06 t08 t10 t11 t12 t13"
for t in t04 t05 t06 t07 t08 t09 t10 t12 t11 t13; do
  log "运行 $t"
  python -m apcs.cli "$t" --config "$RUN_CFG"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    case " $ALLOW_RC1 " in
      *" $t "*)
        if [ "$rc" -eq 1 ]; then
          printf '\033[1;33m[apcs-run][WARN]\033[0m %s 退出码=1（诚实 Gate 结果，继续）\n' "$t"
        elif [ "$rc" -eq 2 ]; then
          fail "$t 被 §72 阻断（退出码=2）：前置 Gate 未过，按规则停止"
        else
          fail "$t 异常退出码=$rc"
        fi
        ;;
      *) fail "$t 退出码=$rc" ;;
    esac
  fi
done

# ── 7. 结果摘要 ───────────────────────────────────────────────────────────────
log "完成。关键结果摘要："
for f in "t01/summary.md" "t04/summary.md" "t05/summary.md" \
         "$RUN_DIR/inject_eval/replacement_score_artifact.json" \
         "t09/summary.md" "t11/summary.md"; do
  p="$RUN_DIR/$f"; [ -f "$p" ] && { echo "───────── $p ─────────"; head -n 24 "$p"; }
done
log "全部产物位于 $RUN_DIR ；论文表格只可引用 measured_task / end_to_end 等级的数字。"
