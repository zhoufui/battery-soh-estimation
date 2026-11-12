$ErrorActionPreference = "Stop"

$env:LOKY_MAX_CPU_COUNT = [string][Environment]::ProcessorCount

python src\train_standard_models.py --test-battery B0018
python src\cross_battery_standard.py

