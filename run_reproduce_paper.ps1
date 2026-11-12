$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot "run_audited_pipeline.ps1") @args
