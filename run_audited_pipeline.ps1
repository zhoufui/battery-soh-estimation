param([switch]$RebuildCalceFeatures)

$ErrorActionPreference = "Stop"
$env:LOKY_MAX_CPU_COUNT = [string][Environment]::ProcessorCount

Push-Location -LiteralPath $PSScriptRoot
try {
    & python -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) { throw "Preprocessing regression tests failed" }
    if ($RebuildCalceFeatures) {
        & python src/prepare_calce_soh.py
        if ($LASTEXITCODE -ne 0) { throw "CALCE preprocessing failed" }
    }
    $steps = @(
        @("src/train_standard_models.py", "--test-battery", "B0018"),
        @("src/cross_battery_standard.py"),
        @("src/calce_validation.py"),
        @("src/representative_domain_adaptation.py"),
        @("src/domain_calibration.py"),
        @("src/audit_soh_features.py"),
        @("src/feature_ablation.py"),
        @("src/make_publication_pngs.py"),
        @("src/validate_audited_outputs.py")
    )
    foreach ($step in $steps) {
        & python @step
        if ($LASTEXITCODE -ne 0) { throw "Failed: python $($step -join ' ')" }
    }
} finally {
    Pop-Location
}
