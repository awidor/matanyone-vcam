param(
    [switch]$Release = $true,
    [string]$OutDir = "",
    [string]$TensorRtRoot = $env:TENSORRT_ROOT,
    [string]$CudaRoot = $env:CUDA_ROOT,
    [switch]$SkipBuild,
    [switch]$Zip
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildType = if ($Release) { "Release" } else { "Debug" }

if (-not $TensorRtRoot) { $TensorRtRoot = "C:\Tools\TensorRT-10.16.1.11" }
if (-not $CudaRoot) { $CudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1" }

$DistRoot = if ($OutDir) { $OutDir } else { Join-Path $Root "dist\matanyone-vcam-win64" }
$ExeSrc = Join-Path $Root "app\target\$($BuildType.ToLower())\matanyone-vcam.exe"
$EnginesSrc = Join-Path $Root "engines\faithful"

if (-not $SkipBuild) {
    & (Join-Path $Root "build.ps1") -Release:$Release
}

if (-not (Test-Path $ExeSrc)) {
    throw "Missing executable: $ExeSrc (build first with .\build.ps1)"
}

Write-Host "Creating bundle at $DistRoot"
if (Test-Path $DistRoot) {
    Remove-Item -Recurse -Force $DistRoot
}
New-Item -ItemType Directory -Path $DistRoot | Out-Null

Copy-Item $ExeSrc (Join-Path $DistRoot "matanyone-vcam.exe")

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
    Copy-Item $src $DistRoot
    Write-Host "  copied $($entry.Name)"
}

if (Test-Path $EnginesSrc) {
    $engineDest = Join-Path $DistRoot "engines\faithful"
    New-Item -ItemType Directory -Path $engineDest -Force | Out-Null
    Copy-Item (Join-Path $EnginesSrc "*.engine") $engineDest
    Write-Host "  copied TensorRT engines"
} else {
    Write-Warning "engines/faithful not found - bundle will not include .engine files"
}

$ReadmeLines = @(
    "MatAnyone Virtual Camera (Windows x64)",
    "",
    "Requirements:",
    "  NVIDIA GPU with up-to-date driver",
    "  Visual C++ 2022 Redistributable (x64)",
    "  Unity Video Capture filter for virtual camera output (recommended)",
    "  Webcam",
    "",
    "Run matanyone-vcam.exe or MatAnyone Virtual Camera.bat",
    "Engines load from .\engines\faithful next to the executable."
)
Set-Content -Path (Join-Path $DistRoot "README.txt") -Value $ReadmeLines -Encoding UTF8

Set-Content -Path (Join-Path $DistRoot "MatAnyone Virtual Camera.bat") -Value "@echo off`r`ncd /d `"%~dp0`"`r`nstart `"`" `"%~dp0matanyone-vcam.exe`" %*`r`n" -Encoding ASCII

if ($Zip) {
    $ZipPath = "$DistRoot.zip"
    if (Test-Path $ZipPath) { Remove-Item -Force $ZipPath }
    Compress-Archive -Path $DistRoot -DestinationPath $ZipPath
    Write-Host "Created $ZipPath"
}

Write-Host "Bundle ready: $DistRoot"
