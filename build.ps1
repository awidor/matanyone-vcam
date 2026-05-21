param(
    [switch]$Release = $true,
    [switch]$Bundle,
    [switch]$Zip
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildType = if ($Release) { "Release" } else { "Debug" }

$TensorRtRoot = if ($env:TENSORRT_ROOT) { $env:TENSORRT_ROOT } else { "C:\Tools\TensorRT-10.16.1.11" }
$CudaRoot = if ($env:CUDA_ROOT) { $env:CUDA_ROOT } else { "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1" }
$env:PATH = "$TensorRtRoot\bin;$CudaRoot\bin\x64;$CudaRoot\bin;" + $env:PATH
$env:TENSORRT_ROOT = $TensorRtRoot
$env:CUDA_ROOT = $CudaRoot

Write-Host "Building workspace ($BuildType)..."
Push-Location $Root
if ($Release) {
    cargo build --release
} else {
    cargo build
}
if ($LASTEXITCODE -ne 0) {
    Pop-Location
    throw "cargo build failed"
}
Pop-Location

$TargetDir = Join-Path $Root "target\$($BuildType.ToLower())"
$AppExe = Join-Path $TargetDir "matanyone-vcam.exe"
if (-not (Test-Path $AppExe)) {
    throw "Missing app binary: $AppExe"
}

Copy-Item $AppExe (Join-Path $Root "matanyone-vcam.exe") -Force
Write-Host "Copied matanyone-vcam.exe -> repo root"

$CopyRuntimeDlls = Join-Path $Root "scripts\copy_runtime_dlls.ps1"
Write-Host "Copying TensorRT/CUDA runtime DLLs -> repo root"
& $CopyRuntimeDlls -DestDir $Root -TensorRtRoot $TensorRtRoot -CudaRoot $CudaRoot

$BinDir = Join-Path $Root "bin"
New-Item -ItemType Directory -Path $BinDir -Force | Out-Null
$DevTools = @(
    "matanyone_core_smoke.exe",
    "matanyone_runner_bench.exe",
    "matanyone_runner_raw.exe",
    "matanyone_trt_bench.exe"
)
foreach ($name in $DevTools) {
    $src = Join-Path $TargetDir $name
    if (Test-Path $src) {
        Copy-Item $src (Join-Path $BinDir $name) -Force
    }
}
Write-Host "Dev tool binaries -> bin\"
Write-Host "Copying TensorRT/CUDA runtime DLLs -> bin\"
& $CopyRuntimeDlls -DestDir $BinDir -TensorRtRoot $TensorRtRoot -CudaRoot $CudaRoot

if ($Bundle) {
    & (Join-Path $Root "bundle.ps1") -Release:$Release -SkipBuild -Zip:$Zip
}

Write-Host "Done."
Write-Host "Run app: .\matanyone-vcam.exe   (runtime DLLs copied beside exe; .\run_vcam.ps1 optional)"
Write-Host "Core smoke: .\bin\matanyone_core_smoke.exe --synthetic engines\faithful"
if ($Bundle) {
    Write-Host "Bundle folder: $Root\dist\matanyone-vcam-win64"
}
