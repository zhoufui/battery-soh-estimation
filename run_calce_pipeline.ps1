$ErrorActionPreference = "Stop"

$env:LOKY_MAX_CPU_COUNT = [string][Environment]::ProcessorCount

python src\prepare_calce_soh.py --batteries CS2_35,CS2_36
python src\calce_validation.py
