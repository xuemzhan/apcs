# ═══════════════════════════════════════════════════════════════════════════════
# APCS 真实 GPU 实验启动器（Windows 开发机侧包装）
#
# 本机为无 GPU 开发环境：本脚本不本地跑实验，只做两件事：
#   1) CPU 冒烟：全量 pytest（必须 160+ 全绿后才允许上 GPU 机）
#   2) 提示如何把仓库同步到 GPU 机器并一键运行
#
# 用法：
#   powershell -File scripts\run_real_gpu.ps1 -SmokeTest     # 本地 CPU 全量测试
#   powershell -File scripts\run_real_gpu.ps1 -Remote user@gpu-host -Repo /path/on/remote
# ═══════════════════════════════════════════════════════════════════════════════
param(
    [switch]$SmokeTest,
    [string]$Remote,
    [string]$Repo
)

if ($SmokeTest) {
    python -m pytest tests/ -q --tb=short
    exit $LASTEXITCODE
}

if (-not $Remote -or -not $Repo) {
    Write-Host @"
GPU 机器上的标准流程（Linux）：

    cd <repo>
    bash scripts/run_real_gpu.sh                     # 标准档
    APCS_CTX4K=1 APCS_CALIB=256 bash scripts/run_real_gpu.sh   # 加 4096 档

可选 rsync 同步（排除大产物）：

    git archive HEAD | ssh $Remote "mkdir -p $Repo && tar -x -C $Repo"

"@
    if ($Remote -and $Repo) {
        Write-Host "如需立即触发远端运行，取消注释下行："
        # ssh $Remote "cd $Repo && nohup bash scripts/run_real_gpu.sh > run.log 2>&1 &"
    }
    exit 0
}
