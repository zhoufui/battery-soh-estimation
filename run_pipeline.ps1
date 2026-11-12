$ErrorActionPreference = "Stop"

$python = "python"

if (-not (Test-Path "data\raw\NASA_Battery_Data_Set.zip") -and -not (Get-ChildItem -Path "data\raw" -Recurse -Filter "*.mat" -ErrorAction SilentlyContinue)) {
    Write-Host "NASA data is missing."
    Write-Host "Download it from:"
    Write-Host "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"
    Write-Host "Save as: data\raw\NASA_Battery_Data_Set.zip"
    exit 1
}

& $python src\prepare_nasa_soh.py
& $python src\train_baseline.py
& $python src\cross_battery_eval.py
& $python src\feature_ablation.py
