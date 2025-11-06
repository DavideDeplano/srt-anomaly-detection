@echo off
setlocal

REM --- CONFIG: path to Miniconda/Anaconda (edit if installed elsewhere) ---
set "CONDA_ROOT=%USERPROFILE%\miniconda3"

REM Check if conda is available in the expected path
if not exist "%CONDA_ROOT%\condabin\conda.bat" (
  echo [ERROR] Conda not found at "%CONDA_ROOT%\condabin\conda.bat".
  echo Please update CONDA_ROOT in this file to match your installation.
  exit /b 1
)

REM Initialize conda commands for batch scripts (avoids PowerShell hooks)
call "%CONDA_ROOT%\condabin\conda.bat" activate

REM Create or update the environment from env.yml
conda env create -f env.yml || conda env update -f env.yml

REM Install the local project in editable mode inside the environment
call conda run -n srt-anom python -m pip install -e .

echo.
echo [OK] Environment ready: srt-anom
echo To run the project without activating the shell: scripts\win\run.bat

endlocal
