param(
    [string]$Baselines = "",
    [double]$Tolerance = 0.02,
    [string]$EngineDir = "engines/faithful"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent
if (-not $Baselines) {
    $Baselines = Join-Path $Root "docs\benchmark_baselines.json"
}

$TensorRtRoot = if ($env:TENSORRT_ROOT) { $env:TENSORRT_ROOT } else { "C:\Tools\TensorRT-10.16.1.11" }
$CudaRoot = if ($env:CUDA_ROOT) { $env:CUDA_ROOT } else { "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1" }
$env:PATH = "$TensorRtRoot\bin;$CudaRoot\bin\x64;$CudaRoot\bin;" + $env:PATH
$env:TENSORRT_ROOT = $TensorRtRoot
$env:CUDA_ROOT = $CudaRoot

Push-Location $Root
try {
    $json = Get-Content $Baselines -Raw | ConvertFrom-Json

    function Parse-Metrics([string]$Output) {
        $metrics = @{}
        foreach ($line in ($Output -split "`n")) {
            if ($line -match '^([^=]+)=(.+)$') {
                $metrics[$Matches[1]] = [double]$Matches[2]
            }
        }
        return $metrics
    }

    function Check-Metric([string]$Name, [double]$Actual, [double]$Baseline) {
        if ($Baseline -eq 0) {
            Write-Warning "Skip $Name baseline is zero"
            return
        }
        $delta = [math]::Abs($Actual - $Baseline) / $Baseline
        $pct = $delta * 100.0
        Write-Host ("  {0}: actual={1:N4} baseline={2:N4} delta={3:P2}" -f $Name, $Actual, $Baseline, $delta)
        if ($delta -gt $Tolerance) {
            throw "Parity failed for ${Name}: $($pct.ToString('N2'))% > $($Tolerance * 100)% tolerance"
        }
    }

    Write-Host "Building release benches..."
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    cargo build --release -p matanyone-core 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        $ErrorActionPreference = $prevEap
        throw "cargo build --release -p matanyone-core failed"
    }
    $ErrorActionPreference = $prevEap

    function Resolve-BenchExe([string]$Name) {
        $bin = Join-Path $Root "bin\$Name"
        if (Test-Path $bin) { return $bin }
        return Join-Path $Root "target\release\$Name"
    }

    $TrtBench = Resolve-BenchExe "matanyone_trt_bench.exe"
    $RunnerBench = Resolve-BenchExe "matanyone_runner_bench.exe"
    $RunnerRaw = Resolve-BenchExe "matanyone_runner_raw.exe"
    if (-not (Test-Path $TrtBench)) {
        throw "Missing $TrtBench (build release benches first)"
    }

    function Invoke-BenchMetrics([string]$Exe, [string[]]$Args) {
        # Cold GPU can read ~2-3x slower than baselines; a short warm-up then one timed run.
        for ($i = 0; $i -lt 2; $i++) { & $Exe @Args 2>&1 | Out-Null }
        $out = & $Exe @Args 2>&1 | Out-String
        return (Parse-Metrics $out), $out
    }

    Write-Host "Running matanyone_trt_bench..."
    $trt, $trtOut = Invoke-BenchMetrics $TrtBench @($EngineDir)
    Write-Host $trtOut
    Check-Metric "normal_mean_ms" $trt["normal_mean_ms"] $json.matanyone_trt_bench.normal_mean_ms
    Check-Metric "normal_p99_ms" $trt["normal_p99_ms"] $json.matanyone_trt_bench.normal_p99_ms
    Check-Metric "memory_update_mean_ms" $trt["memory_update_mean_ms"] $json.matanyone_trt_bench.memory_update_mean_ms
    Check-Metric "memory_update_p99_ms" $trt["memory_update_p99_ms"] $json.matanyone_trt_bench.memory_update_p99_ms

    Write-Host "Running matanyone_runner_bench..."
    $runner, $runnerOut = Invoke-BenchMetrics $RunnerBench @($EngineDir)
    Write-Host $runnerOut
    Check-Metric "mixed_mean_ms" $runner["mixed_mean_ms"] $json.matanyone_runner_bench.mixed_mean_ms
    Check-Metric "mixed_p99_ms" $runner["mixed_p99_ms"] $json.matanyone_runner_bench.mixed_p99_ms

    if (Test-Path (Join-Path $Root "raw_sample\mask.f32")) {
        Write-Host "Running matanyone_runner_raw..."
        $rawOut = & $RunnerRaw raw_sample $EngineDir output/raw_alpha 8 2>&1 | Out-String
        Write-Host $rawOut
        $raw = Parse-Metrics $rawOut
        if ($raw.ContainsKey("mean_ms")) {
            Check-Metric "mean_ms" $raw["mean_ms"] $json.matanyone_runner_raw.mean_ms
            Check-Metric "p99_ms" $raw["p99_ms"] $json.matanyone_runner_raw.p99_ms
        }
    } else {
        Write-Warning "raw_sample not found; skipping matanyone_runner_raw parity"
    }

    Write-Host "All parity checks passed within $([int]($Tolerance * 100))% tolerance."
}
finally {
    Pop-Location
}
