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

Write-Host "Building matanyone_core ($BuildType)..."
& cmd.exe /c "call `"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat`" && cmake -S `"$Root\core`" -B `"$Root\build\core`" -G `"Visual Studio 17 2022`" -A x64 && cmake --build `"$Root\build\core`" --config $BuildType"

Write-Host "Building matanyone-vcam ($BuildType)..."
Push-Location "$Root\app"
if ($Release) {
    cargo build --release
} else {
    cargo build
}
Pop-Location

if ($Bundle) {
    & (Join-Path $Root "bundle.ps1") -Release:$Release -SkipBuild -Zip:$Zip
}

Write-Host "Done."
Write-Host "Core smoke test: $Root\build\core\$BuildType\matanyone_core_smoke.exe --synthetic engines\faithful"
Write-Host "App binary:      $Root\app\target\$($BuildType.ToLower())\matanyone-vcam.exe"
if ($Bundle) {
    Write-Host "Bundle folder:   $Root\dist\matanyone-vcam-win64"
}
