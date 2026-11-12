$ErrorActionPreference = "Stop"

$env:LOKY_MAX_CPU_COUNT = [string][Environment]::ProcessorCount

python src\domain_calibration.py
python src\representative_domain_adaptation.py

