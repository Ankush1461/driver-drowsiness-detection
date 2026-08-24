@echo off
REM DriveSafe AI - Keep-Alive Wrapper for Windows Task Scheduler
REM This batch file runs the Python keep-alive script in single-ping mode
REM Schedule this to run every 10 minutes via Task Scheduler

cd /d "%~dp0"
python keep_alive.py --once
