param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8000,
    [switch]$Reload
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$argsList = @("api.main:app", "--host", $HostAddress, "--port", $Port)
if ($Reload) { $argsList += "--reload" }
uvicorn @argsList
