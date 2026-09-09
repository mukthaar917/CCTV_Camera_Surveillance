param([string]$Config = "config\training.yaml")
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
python -m src.training.train --config $Config
if ($LASTEXITCODE -ne 0) { throw "Training failed." }
