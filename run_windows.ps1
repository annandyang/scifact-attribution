param(
    [ValidateSet("Core", "Smoke", "Dev", "Test", "Ablation", "Demo")]
    [string]$Stage = "Dev",
    [string]$Python = "",
    [string]$Seeds = "0",
    [switch]$SkipE1,
    [switch]$WithPerturbation,
    [switch]$FullPrecisionGrid,
    [int]$Port = 7860
)

$ErrorActionPreference = "Stop"
if (-not $Python) { $Python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe' }
if ($Python -match '[\\/]') { $Python = (Resolve-Path -LiteralPath $Python).Path }
if (-not $env:HF_HOME) { $env:HF_HOME = Join-Path $PSScriptRoot '.cache\huggingface' }
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:HF_HUB_DOWNLOAD_TIMEOUT = '180'
$env:GRADIO_ANALYTICS_ENABLED = 'False'
$env:PYTHONUTF8 = '1'

function Run {
    & $Python @args
    if ($LASTEXITCODE -ne 0) { throw "Failed: $($args -join ' ')" }
}

Push-Location $PSScriptRoot
try {
    if ($Stage -eq "Demo") {
        Run scripts/00_check_env.py --stage demo
        Run app/gradio_app.py --split dev --port $Port
        return
    }
    if ($Stage -eq "Core") {
        Run scripts/00_check_env.py
        Run -m unittest discover -s tests
        Run scripts/01_prepare_data.py
        Run scripts/02_build_index.py --retriever bm25
        Run scripts/03_retrieval_study.py --bm25-only
        Run scripts/04_run_condition.py --condition c2 --dry-run --limit 10
        Run scripts/07_shortcut_audit.py
        return
    }
    Run scripts/00_check_env.py --stage gpu
    if ($Stage -in 'Smoke','Dev') {
        if (-not (Test-Path 'data/processed/verify_report.json')) { Run scripts/01_prepare_data.py }
        if (-not (Test-Path 'data/indexes/bm25_scifact.pkl')) { Run scripts/02_build_index.py --retriever bm25 }
        if (-not (Test-Path 'data/indexes/dense_scifact.npy')) { Run scripts/02_build_index.py --retriever dense }
    }
    if ($Stage -eq "Smoke") {
        foreach ($condition in 'c0','c1','c2') {
            Run scripts/04_run_condition.py --condition $condition --limit 5
        }
        Run scripts/05_train_lora.py --condition c3 --smoke
        foreach ($condition in 'c3','c4') {
            Run scripts/04_run_condition.py --condition $condition --seeds 0 --smoke-adapter --limit 5
        }
        return
    }
    if ($Stage -eq "Ablation") {
        $c3 = 'outputs/predictions/c3_dev'
        if (-not (Test-Path "$c3.jsonl") -or -not (Test-Path "$c3.meta.json")) {
            throw "Run the Dev stage with matching seeds before Ablation."
        }
        $baseline = Get-Content "$c3.meta.json" -Raw | ConvertFrom-Json
        $baselineSeeds = ($baseline.seeds | Sort-Object) -join ','
        $requestedSeeds = (($Seeds -split ',' | ForEach-Object { [int]$_ }) | Sort-Object) -join ','
        if ($baselineSeeds -ne $requestedSeeds -or $baseline.limit -ne 0 -or $baseline.dry_run -or $baseline.smoke_adapter) {
            throw "Ablation needs full C3 predictions with the same seeds. Found: $baselineSeeds."
        }
        foreach ($rank in 8,32) {
            Run scripts/05_train_lora.py --condition c3 --rank $rank --seeds $Seeds
            Run scripts/04_run_condition.py --condition c3 --seeds $Seeds --adapter-base "outputs/adapters/lora_r16_r$($rank)_qlora" --out "outputs/predictions/rank$($rank)_dev.jsonl"
        }
        $conditions = 'rank8,c3,rank32'
        if ($FullPrecisionGrid) {
            foreach ($rank in 8,16,32) {
                Run scripts/05_train_lora.py --condition c3 --rank $rank --seeds $Seeds --precision bf16_lora
                Run scripts/04_run_condition.py --condition c3 --seeds $Seeds --precision bf16_lora --adapter-base "outputs/adapters/lora_r16_r${rank}_bf16_lora" --out "outputs/predictions/bf16_r${rank}_dev.jsonl"
            }
            $conditions += ',bf16_r8,bf16_r16,bf16_r32'
        }
        Run scripts/08_report_tables.py --split dev --conditions $conditions --tag ablation
        return
    }
    $split = $Stage.ToLowerInvariant()
    if ($Stage -eq "Dev" -and -not $SkipE1) {
        Run scripts/03_retrieval_study.py --with-rerank
    }
    foreach ($condition in 'c0','c1','c2') {
        Run scripts/04_run_condition.py --condition $condition --split $split
    }
    if ($Stage -eq "Dev") { Run scripts/05_train_lora.py --condition c3 --seeds $Seeds }
    foreach ($condition in 'c3','c4') {
        Run scripts/04_run_condition.py --condition $condition --split $split --seeds $Seeds
    }
    Run scripts/06_calibrate.py --split $split
    if ($Stage -eq "Dev") {
        Run scripts/08_report_tables.py --split dev --conditions "c0,c1,c2,c3,c4,c5"
        Run scripts/05_train_lora.py --condition c3_prime --seeds 0
        Run scripts/04_run_condition.py --condition c3_prime --seeds 0
        Run scripts/07_shortcut_audit.py
        Run scripts/10_error_review.py --seed 0
        if ($WithPerturbation) { Run scripts/09_evidence_perturb.py --seed 0 }
    }
}
finally {
    Pop-Location
}
