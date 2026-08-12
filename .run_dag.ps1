$env:PYTHONIOENCODING='utf-8'
$tasks = @('t00','t01','t02','t03','t04','t05','t06','t07','t08','t09','t10','t12','t11','t13')
$ok = $true
foreach ($t in $tasks) {
    Write-Host "===== $t ====="
    $proc = Start-Process -FilePath python -ArgumentList '-m','apcs.cli',$t,'--config','configs/pair_qwen3.yaml' -NoNewWindow -Wait -PassThru -RedirectStandardOutput "reports\.${t}.dag.out" -RedirectStandardError "reports\.${t}.dag.err"
    $tail = Get-Content "reports\.${t}.dag.out" -ErrorAction SilentlyContinue | Select-Object -Last 1
    Write-Host "  exit=$($proc.ExitCode) last=$tail"
    if ($proc.ExitCode -ne 0) { $ok = $false }
}
Write-Host "OK=$ok"
