param(
    [string]$Data = "datasets\merged_dataset\data.yaml",
    [string]$Weights = ""
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
python -m src.data.validate_dataset --data $Data
if ($LASTEXITCODE -ne 0) { throw "Dataset validation failed." }
if ($Weights -ne "") {
    python -m src.evaluation.evaluate --weights $Weights --data $Data --split val
    if ($LASTEXITCODE -ne 0) { throw "Model evaluation failed." }
}
