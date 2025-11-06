@echo off
setlocal

REM --- CONFIG: path to Miniconda/Anaconda ---
set "CONDA_ROOT=%USERPROFILE%\miniconda3"

REM Run the main module inside the 'srt-anom' environment
REM %* passes any additional arguments to the Python program
call "%CONDA_ROOT%\condabin\conda.bat" run -n srt-anom python -m src.srtad.main %*

endlocal
