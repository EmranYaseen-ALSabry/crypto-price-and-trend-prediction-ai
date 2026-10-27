@echo off
setlocal
if exist ".\.venv\Scripts\python.exe" (
    set "PY_EXE=.\.venv\Scripts\python.exe"
) else if exist "..\..\..\envairoment\.venv\Scripts\python.exe" (
    set "PY_EXE=..\..\..\envairoment\.venv\Scripts\python.exe"
) else (
    set "PY_EXE=python"
)
"%PY_EXE%" predict.py %*
endlocal
