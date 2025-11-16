@echo off
setlocal

REM --- CONFIG: path to Miniconda/Anaconda ---
set "CONDA_ROOT=%USERPROFILE%\miniconda3"

REM Activate environment (mandatory for interactive input)
call "%CONDA_ROOT%\condabin\conda.bat" activate srt-anom

REM Run main module normally (interactive OK)
python -m src.srtad.main %*

endlocal
