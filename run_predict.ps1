param(
    [string]$symbol = "BTCUSDT",
    [string]$interval = "1h",
    [int]$horizon = 1
)

& "..\..\..\envairoment\.venv\Scripts\python.exe" predict.py --symbol $symbol --interval $interval --horizon $horizon
