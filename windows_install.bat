@echo off
echo Installing fairseq with CUDA support on Windows

REM Set CUDA environment variables
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4
set CUDA_PATH=%CUDA_HOME%
set PATH=%CUDA_HOME%\bin;%PATH%

REM Print environment for debugging
echo CUDA_HOME: %CUDA_HOME%
echo CUDA_PATH: %CUDA_PATH%

REM Install fairseq
pip install -e .

pause 