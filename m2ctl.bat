@echo off
rem M2 0.2.1 Windows launcher, equivalent of the macOS ./m2ctl script.
rem Usage: m2ctl.bat <any m2.cli subcommand, e.g. health / run / serve --port 8765>
setlocal
set "ROOT=%~dp0"
cd /d "%ROOT%"

set "HF_HOME=%ROOT%.hf-cache"
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"

set "PY=%ROOT%.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo [m2ctl] .venv not found, falling back to system python.
  echo [m2ctl] If imports fail, follow the install steps in README.md.
  set "PY=python"
)

"%PY%" -m m2.cli %*
exit /b %ERRORLEVEL%
