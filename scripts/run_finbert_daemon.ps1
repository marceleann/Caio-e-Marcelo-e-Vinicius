# =============================================================================
# Daemon do scorer FinBERT — roda ATE A FILA ACABAR, com checkpoints duplos.
# Registrado como tarefa agendada do Windows (logon): independe do Claude Code
# e religa sozinho se o PC reiniciar. Aprovado pelo Marcelo em 17/07/2026.
#
# Laco: pontua por 6h -> commita e pusha os shards novos p/ o GitHub -> repete.
# Para quando uma rodada nao produz nenhuma call nova (fila vazia).
# Checkpoint 1: 1 parquet por call (atomico, no proprio scorer).
# Checkpoint 2: push p/ GitHub a cada ciclo de 6h.
# Log: logs\finbert_daemon.log
# =============================================================================
$proj = "C:\Users\Marcelo\Desafio Itaú QAI 26"
Set-Location $proj
New-Item -ItemType Directory -Force "$proj\logs" | Out-Null
$log = "$proj\logs\finbert_daemon.log"
$shards = "$proj\data\interim\sentence_scores_shards"

# instancia unica: se ja ha um python rodando o scorer, nao duplica
$dup = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
       Where-Object { $_.CommandLine -match "sp500_score_finbert" }
if ($dup) {
    Add-Content $log "$(Get-Date -f s) daemon: scorer ja em execucao (PID $($dup.ProcessId)); saindo."
    exit 0
}

Add-Content $log "$(Get-Date -f s) daemon: iniciando laco."
while ($true) {
    $before = (Get-ChildItem $shards -Filter "call_*.parquet" -ErrorAction SilentlyContinue).Count
    Add-Content $log "$(Get-Date -f s) ciclo: $before/33362 shards; rodando 6h..."
    cmd /c "python scripts\sp500_score_finbert.py 6 >> `"$log`" 2>&1"
    $after = (Get-ChildItem $shards -Filter "call_*.parquet" -ErrorAction SilentlyContinue).Count

    # checkpoint remoto: TD_FinBERT atualizada + push
    cmd /c "python scripts\sp500_td_finbert.py >> `"$log`" 2>&1"
    git add data/interim/sentence_scores_shards data/interim/sp500/tone_distance_finbert.parquet 2>$null
    git commit -m "checkpoint FinBERT automatico: $after/33362 calls" 2>$null
    git push origin Projeto_Caio_Marcelo 2>$null
    Add-Content $log "$(Get-Date -f s) checkpoint: $after/33362 shards; push feito."

    if ($after -eq $before) {
        Add-Content $log "$(Get-Date -f s) daemon: rodada sem calls novas — fila concluida (ou erro persistente; ver log acima). FIM."
        break
    }
}
