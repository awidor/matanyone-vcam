param(
    [Parameter(Mandatory = $true)]
    [string]$DestDir,
    [string]$TensorRtRoot = "",
    [string]$CudaRoot = ""
)

$ErrorActionPreference = "Stop"

if (-not $TensorRtRoot) {
    $TensorRtRoot = if ($env:TENSORRT_ROOT) { $env:TENSORRT_ROOT } else { "C:\Tools\TensorRT-10.16.1.11" }
}
if (-not $CudaRoot) {
    $CudaRoot = if ($env:CUDA_ROOT) { $env:CUDA_ROOT } else { "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1" }
}

New-Item -ItemType Directory -Path $DestDir -Force | Out-Null

$TrtBin = Join-Path $TensorRtRoot "bin"
$CudaBin = Join-Path $CudaRoot "bin\x64"
$RuntimeDlls = @(
    @{ Name = "nvinfer_10.dll"; Source = $TrtBin },
    @{ Name = "nvinfer_plugin_10.dll"; Source = $TrtBin },
    @{ Name = "nvinfer_dispatch_10.dll"; Source = $TrtBin },
    @{ Name = "cudart64_13.dll"; Source = $CudaBin }
)

foreach ($entry in $RuntimeDlls) {
    $src = Join-Path $entry.Source $entry.Name
    if (-not (Test-Path $src)) {
        throw "Required runtime DLL not found: $src"
    }
    Copy-Item $src $DestDir -Force
    Write-Host "  copied $($entry.Name) -> $DestDir"
}
