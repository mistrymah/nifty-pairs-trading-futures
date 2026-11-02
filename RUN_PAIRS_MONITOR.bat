@echo off
title Nifty Live Pairs Trading Monitor
cd /d "%~dp0"
echo Starting Live Pairs Trading Monitor (5m Z-Score)...

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" live_pairs_monitor.py
) else if exist "c:\Users\admin\Desktop\Trading_Workspace\Env antigravity\.venv\Scripts\python.exe" (
    "c:\Users\admin\Desktop\Trading_Workspace\Env antigravity\.venv\Scripts\python.exe" live_pairs_monitor.py
) else (
    python live_pairs_monitor.py
)
pause
