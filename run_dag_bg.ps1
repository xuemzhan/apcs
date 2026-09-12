$env:PYTHONIOENCODING='utf-8'
Set-Location E:\Workspace\KVCache
# Cfg-driven small runs to avoid OOM; user can override via cfg if they have more RAM
$tasks = @('t00','t01','t02','t03','t04','t05','t06','t07','t08','t09','t10','t12','t11','t13')
foreach ($t in $tasks) {
    Write-Host "===== $t ====="
    $proc = Start-Process -FilePath python -ArgumentList @('-m','apcs.cli',$t,'--config','configs/pair_qwen3.yaml') -NoNewWindow -Wait -PassThru -RedirectStandardOutput "reports\.${t}.dag.out" -RedirectStandardError "reports\.${t}.dag.err"
    $result = (Get-Content "reports\.${t}.dag.out" -ErrorAction SilentlyContinue | Select-Object -Last 1)
    if ($proc.ExitCode -eq 0) {
        Write-Host "  OK $t :: $result"
    } else {
        Write-Host "  FAIL $t (exit=$($proc.ExitCode)) :: $result"
    }
}
Write-Host "DAG complete"
