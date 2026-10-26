param(
    [string]$symbol = "BTCUSDT",
    [int]$horizon = 1
)

& "..\..\..\envairoment\.venv\Scripts\python.exe" predict.py --symbol $symbol --horizon $horizon
