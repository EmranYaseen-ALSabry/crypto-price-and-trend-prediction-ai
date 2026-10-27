param(
    [string]$symbol = "BTCUSDT",
    [string]$interval = "1h",
    [int]$horizon = 1,
    [switch]$trainIfMissing,
    [switch]$noCharts
)

$pyExe = "python"
if (Test-Path ".\.venv\Scripts\python.exe") {
    $pyExe = ".\.venv\Scripts\python.exe"
} elseif (Test-Path "..\..\..\envairoment\.venv\Scripts\python.exe") {
    $pyExe = "..\..\..\envairoment\.venv\Scripts\python.exe"
}

$cmdArgs = @("predict.py", "--symbol", $symbol, "--interval", $interval, "--horizon", $horizon)
if ($trainIfMissing) {
    $cmdArgs += "--train-if-missing"
}
if ($noCharts) {
    $cmdArgs += "--no-charts"
}

& $pyExe $cmdArgs
