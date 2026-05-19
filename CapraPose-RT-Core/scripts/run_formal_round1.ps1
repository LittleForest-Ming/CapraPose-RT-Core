param(
    [string]$TrainPython = "D:\Anaconda\envs\pytorch\python.exe",
    [string]$ToolsPython = "D:\Anaconda\python.exe"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$RunRoot = Join-Path $RepoRoot "outputs\experiments"
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
$LogPath = Join-Path $RunRoot "formal_round1_run.log"
$StatusPath = Join-Path $RunRoot "formal_round1_run_status.json"
$TableDir = Join-Path $RunRoot "formal_round1_tables\full_or_available"

function Write-RunLog([string]$Message) {
    $Line = "{0} | {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    $Line | Tee-Object -FilePath $LogPath -Append | Out-Null
}

function Update-RunStatus([array]$Experiments, [string]$OverallStatus) {
    $Payload = [ordered]@{
        updated_at = (Get-Date).ToString("s")
        overall_status = $OverallStatus
        experiments = $Experiments
    }
    $Payload | ConvertTo-Json -Depth 8 | Set-Content -Path $StatusPath -Encoding UTF8
}

function Get-ExperimentSummary([string]$SummaryPath) {
    if (-not (Test-Path $SummaryPath)) {
        return $null
    }
    return Get-Content -Path $SummaryPath -Raw | ConvertFrom-Json
}

function Test-TrainingComplete([string]$SummaryPath) {
    $Summary = Get-ExperimentSummary -SummaryPath $SummaryPath
    if ($null -eq $Summary) {
        return $false
    }
    return ($Summary.status -eq "completed" -and [int]$Summary.current_epoch -ge [int]$Summary.epochs_total)
}

function Archive-IncompleteWorkDir([string]$WorkDir) {
    if (-not (Test-Path $WorkDir)) {
        return
    }
    $Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $ArchivedPath = "{0}_interrupted_{1}" -f $WorkDir, $Timestamp
    Move-Item -LiteralPath $WorkDir -Destination $ArchivedPath
    Write-RunLog ("Archived incomplete work directory to {0}" -f $ArchivedPath)
}

$DatasetRoot = if ($env:CAPRAPOSE_DATASET_ROOT) {
    $env:CAPRAPOSE_DATASET_ROOT
} else {
    "F:\GZM\Dataset_Goat\Pose_Estimation\Dataset1"
}
$ValAnn = Join-Path $RepoRoot "data\normalized_annotations\Dataset1\annotations\val.json"
$ValImageRoot = Join-Path $DatasetRoot "images\train"

$Experiments = @(
    [ordered]@{
        name = "baseline"
        experiment_name = "baseline_rtmpose_m_goat17"
        config = "configs\baseline_rtmpose.py"
        work_dir = "outputs\experiments\baseline_rtmpose_m_goat17"
        status = "pending"
    },
    [ordered]@{
        name = "decoder"
        experiment_name = "caprapose_rt_decoder_goat17"
        config = "configs\caprapose_rt_decoder.py"
        work_dir = "outputs\experiments\caprapose_rt_decoder_goat17"
        status = "pending"
    },
    [ordered]@{
        name = "decoder_refine"
        experiment_name = "caprapose_rt_decoder_refine_goat17"
        config = "configs\caprapose_rt_decoder_refine.py"
        work_dir = "outputs\experiments\caprapose_rt_decoder_refine_goat17"
        status = "pending"
    },
    [ordered]@{
        name = "full"
        experiment_name = "caprapose_rt_full_goat17"
        config = "configs\caprapose_rt.py"
        work_dir = "outputs\experiments\caprapose_rt_full_goat17"
        status = "pending"
    }
)

Update-RunStatus -Experiments $Experiments -OverallStatus "running"
Write-RunLog "Starting formal round-1 experiment queue."

foreach ($Experiment in $Experiments) {
    $WorkDir = Join-Path $RepoRoot $Experiment.work_dir
    $CheckpointPath = Join-Path $WorkDir "checkpoints\best.pth"
    $EvalDir = Join-Path $WorkDir "evaluation\val_best"
    $PredictionPath = Join-Path $EvalDir "predictions.json"
    $SummaryPath = Join-Path $EvalDir "summary.json"
    $HistoryPath = Join-Path $WorkDir "metrics\metrics_history.jsonl"
    $QualitativeDir = Join-Path $WorkDir "qualitative\val_best"
    $ExperimentSummaryPath = Join-Path $WorkDir "experiment_summary.json"
    $TrainingComplete = Test-TrainingComplete -SummaryPath $ExperimentSummaryPath

    try {
        if ($TrainingComplete -and (Test-Path $SummaryPath) -and (Test-Path $PredictionPath)) {
            $Experiment.status = "completed"
            $Experiment.best_checkpoint = $CheckpointPath
            $Experiment.eval_summary = $SummaryPath
            Write-RunLog ("Skipping completed experiment {0}." -f $Experiment.experiment_name)
            Update-RunStatus -Experiments $Experiments -OverallStatus "running"
            continue
        }

        if ((-not $TrainingComplete) -and (Test-Path $ExperimentSummaryPath)) {
            Archive-IncompleteWorkDir -WorkDir $WorkDir
            $CheckpointPath = Join-Path $WorkDir "checkpoints\best.pth"
            $EvalDir = Join-Path $WorkDir "evaluation\val_best"
            $PredictionPath = Join-Path $EvalDir "predictions.json"
            $SummaryPath = Join-Path $EvalDir "summary.json"
            $HistoryPath = Join-Path $WorkDir "metrics\metrics_history.jsonl"
            $QualitativeDir = Join-Path $WorkDir "qualitative\val_best"
        }

        if (-not $TrainingComplete) {
            $Experiment.status = "training"
            Update-RunStatus -Experiments $Experiments -OverallStatus "running"
            Write-RunLog ("Training {0}" -f $Experiment.experiment_name)
            & $TrainPython train.py --config $Experiment.config --work-dir $Experiment.work_dir --train-num-workers 0 --eval-num-workers 0
            if ($LASTEXITCODE -ne 0) {
                throw ("Training failed with exit code {0}" -f $LASTEXITCODE)
            }
            $TrainingComplete = Test-TrainingComplete -SummaryPath $ExperimentSummaryPath
        }

        $Experiment.status = "evaluating"
        Update-RunStatus -Experiments $Experiments -OverallStatus "running"
        Write-RunLog ("Evaluating {0}" -f $Experiment.experiment_name)
        & $TrainPython eval.py --config $Experiment.config --work-dir $Experiment.work_dir --checkpoint $CheckpointPath --split val
        if ($LASTEXITCODE -ne 0) {
            throw ("Evaluation failed with exit code {0}" -f $LASTEXITCODE)
        }

        $Experiment.status = "qualitative"
        Update-RunStatus -Experiments $Experiments -OverallStatus "running"
        Write-RunLog ("Exporting qualitative figures for {0}" -f $Experiment.experiment_name)
        & $ToolsPython tools\export_experiment_qualitative.py --config $Experiment.config --prediction-json $PredictionPath --split val --output-dir $QualitativeDir
        if ($LASTEXITCODE -ne 0) {
            throw ("Qualitative export failed with exit code {0}" -f $LASTEXITCODE)
        }

        $Experiment.status = "completed"
        $Experiment.best_checkpoint = $CheckpointPath
        $Experiment.eval_summary = $SummaryPath
        $Experiment.history = $HistoryPath
        Update-RunStatus -Experiments $Experiments -OverallStatus "running"
    } catch {
        $Experiment.status = "failed"
        $Experiment.error = $_.Exception.Message
        Write-RunLog ("Experiment {0} failed: {1}" -f $Experiment.experiment_name, $_.Exception.Message)
        Update-RunStatus -Experiments $Experiments -OverallStatus "failed"
    }
}

$CompletedExperiments = @()
foreach ($Experiment in $Experiments) {
    $SummaryPath = Join-Path $RepoRoot ($Experiment.work_dir + "\evaluation\val_best\summary.json")
    $HistoryPath = Join-Path $RepoRoot ($Experiment.work_dir + "\metrics\metrics_history.jsonl")
    if (Test-Path $SummaryPath) {
        $CompletedExperiments += [ordered]@{
            name = $Experiment.name
            config = $Experiment.config
            summary = $SummaryPath
            history = $HistoryPath
        }
    }
}

if ($CompletedExperiments.Count -gt 0) {
    Write-RunLog "Exporting available experiment tables."
    $ArgumentList = @("tools\export_result_tables.py")
    foreach ($Experiment in $CompletedExperiments) {
        $Spec = "{0}::{1}::{2}::{3}" -f $Experiment.name, $Experiment.config, $Experiment.summary, $Experiment.history
        $ArgumentList += "--experiment"
        $ArgumentList += $Spec
    }
    $ArgumentList += "--output-dir"
    $ArgumentList += $TableDir
    & $TrainPython @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        Write-RunLog ("Table export failed with exit code {0}" -f $LASTEXITCODE)
    }
}

$BaselinePredictions = Join-Path $RepoRoot "outputs\experiments\baseline_rtmpose_m_goat17\evaluation\val_best\predictions.json"
$FullPredictions = Join-Path $RepoRoot "outputs\experiments\caprapose_rt_full_goat17\evaluation\val_best\predictions.json"
$ComparisonOutputDir = Join-Path $RepoRoot "outputs\paper_figures\qualitative\formal_round1_baseline_vs_full"
if ((Test-Path $BaselinePredictions) -and (Test-Path $FullPredictions) -and (Test-Path $ValAnn)) {
    Write-RunLog "Exporting baseline-vs-full qualitative comparison figures."
    & $ToolsPython tools\generate_qualitative_figures.py --ann-file $ValAnn --image-root $ValImageRoot --baseline-predictions $BaselinePredictions --full-predictions $FullPredictions --num-samples 6 --seed 0 --output-dir $ComparisonOutputDir
    if ($LASTEXITCODE -ne 0) {
        Write-RunLog ("Baseline-vs-full comparison export failed with exit code {0}" -f $LASTEXITCODE)
    }
}

$HasFailures = ($Experiments | Where-Object { $_.status -eq "failed" }).Count -gt 0
$OverallStatus = if ($HasFailures) { "completed_with_failures" } else { "completed" }
Update-RunStatus -Experiments $Experiments -OverallStatus $OverallStatus
Write-RunLog ("Formal round-1 queue finished with status: {0}" -f $OverallStatus)
