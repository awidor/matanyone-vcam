$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Exe = Join-Path $Root "matanyone-vcam.exe"

$TensorRtRoot = if ($env:TENSORRT_ROOT) { $env:TENSORRT_ROOT } else { "C:\Tools\TensorRT-10.16.1.11" }
$CudaRoot = if ($env:CUDA_ROOT) { $env:CUDA_ROOT } else { "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1" }
$env:PATH = "$TensorRtRoot\bin;$CudaRoot\bin\x64;$CudaRoot\bin;" + $env:PATH
$env:TENSORRT_ROOT = $TensorRtRoot
$env:CUDA_ROOT = $CudaRoot

if (-not (Test-Path $Exe)) {
    Write-Host "matanyone-vcam.exe not found; building..."
    & (Join-Path $Root "build.ps1") -Release
}

Push-Location $Root
try {
    & $Exe @args
} finally {
    Pop-Location
}
