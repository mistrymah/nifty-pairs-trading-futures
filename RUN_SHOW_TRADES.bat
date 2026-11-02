@echo off
title Completed Pairs Paper Trades Log
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" show_pairs_trades.py
) else if exist "c:\Users\admin\Desktop\Trading_Workspace\Env antigravity\.venv\Scripts\python.exe" (
    "c:\Users\admin\Desktop\Trading_Workspace\Env antigravity\.venv\Scripts\python.exe" show_pairs_trades.py
) else (
    python show_pairs_trades.py
)
pause
