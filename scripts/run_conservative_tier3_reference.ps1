param(
    [string]$Device = ""
)
$ErrorActionPreference = "Stop"

$cmake = Get-Command cmake -ErrorAction SilentlyContinue
if (-not $cmake) { throw "cmake wasn't found on PATH." }
$ctest = Get-Command ctest -ErrorAction SilentlyContinue
if (-not $ctest) { throw "ctest wasn't found on PATH." }

function Invoke-Native {
    param(
        [string]$Exe,
        [string[]]$Arguments,
        [string]$FailureLabel
    )
    $prevPref = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Exe @Arguments
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prevPref
    }
    if ($code -ne 0) { throw "$FailureLabel ($code)." }
}

$root = Split-Path $PSScriptRoot -Parent
$runtime = Join-Path $root "runtime"
$build = Join-Path $runtime "build\tier3-reference"

Invoke-Native $cmake.Source @(
    "-S", $runtime,
    "-B", $build,
    "-G", "Visual Studio 17 2022",
    "-A", "x64",
    "-DLEONIDA_VULKAN_HEADERS_FETCH=ON"
) "CMake configure failed"

Invoke-Native $cmake.Source @(
    "--build", $build,
    "--config", "RelWithDebInfo",
    "--target", "d3d12_conservative_tier3_reference_test"
) "Building the Tier 3 reference test failed"

$hadPreviousFilter = Test-Path Env:LEONIDA_VK_DEVICE
$previousFilter = $env:LEONIDA_VK_DEVICE
try {
    if ([string]::IsNullOrWhiteSpace($Device)) {
        if (Test-Path Env:LEONIDA_VK_DEVICE) { Remove-Item Env:LEONIDA_VK_DEVICE }
    } else {
        $env:LEONIDA_VK_DEVICE = $Device.Trim()
    }
    Invoke-Native $ctest.Source @(
        "--test-dir", $build,
        "-C", "RelWithDebInfo",
        "--no-tests=error",
        "-R", "^d3d12_conservative_tier3_reference$",
        "--output-on-failure",
        "-V"
    ) "The Tier 3 reference test failed"
} finally {
    if (-not $hadPreviousFilter) {
        if (Test-Path Env:LEONIDA_VK_DEVICE) { Remove-Item Env:LEONIDA_VK_DEVICE }
    } else {
        $env:LEONIDA_VK_DEVICE = $previousFilter
    }
}
