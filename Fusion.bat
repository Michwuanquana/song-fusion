@echo off
rem Song Fusion UI -> http://localhost:7861
cd /d "%~dp0"
set PYTHONPATH=%~dp0
set PYTHONUTF8=1
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
start "" http://localhost:7861
"%~dp0vendor\ACE-Step-1.5\.venv\Scripts\python.exe" app.py
pause
