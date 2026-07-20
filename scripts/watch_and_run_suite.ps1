# =============================================================================
# Vigia noturno: espera a fila ELEGIVEL do FinBERT zerar e dispara a suite
# final LM x FinBERT, commit e push. Roda uma vez e sai (marker evita repeticao).
# Gatilho: faltantes == 0, OU faltantes < 30 e estavel por 3 checagens seguidas
# (residuo de calls sem falas elegiveis, que nunca geram shard).
# Log: logs\watcher_suite.log
# =============================================================================
$proj = "C:\Users\Marcelo\Desafio Itaú QAI 26"
Set-Location $proj
$log = "$proj\logs\watcher_suite.log"
$marker = "$proj\logs\suite_done.marker"
$py = "C:\Users\Marcelo\venv_qai\Scripts\python.exe"

if (Test-Path $marker) {
    Add-Content $log "$(Get-Date -f s) vigia: marker presente; suite ja rodou. Saindo."
    exit 0
}
Add-Content $log "$(Get-Date -f s) vigia: iniciado."
$prev = -1; $stable = 0
while ($true) {
    $missing = [int](& $py "scripts\check_eligible_missing.py" 2>$null)
    Add-Content $log "$(Get-Date -f s) vigia: faltam $missing elegiveis."
    if ($missing -eq 0) { break }
    if ($missing -lt 30 -and $missing -eq $prev) {
        $stable++
        if ($stable -ge 3) {
            Add-Content $log "$(Get-Date -f s) vigia: $missing residuais estaveis (sem falas elegiveis); prosseguindo."
            break
        }
    } else { $stable = 0 }
    $prev = $missing
    Start-Sleep -Seconds 600
}
Add-Content $log "$(Get-Date -f s) vigia: fila elegivel fechada. Rodando TD_FinBERT + suite..."
cmd /c "`"$py`" scripts\sp500_td_finbert.py >> `"$log`" 2>&1"
cmd /c "`"$py`" scripts\sp500_finbert_suite.py >> `"$log`" 2>&1"
git add docs/RESULTADOS_FINBERT_AUTORUN.txt data/interim/sp500/tone_distance_finbert.parquet 2>$null
git commit -m "Suite final LM x FinBERT (auto: fila elegivel concluida)" 2>$null
git push origin Projeto_Caio_Marcelo 2>$null
New-Item -ItemType File $marker -Force | Out-Null
Add-Content $log "$(Get-Date -f s) vigia: suite concluida, resultados commitados e pushados. FIM."

