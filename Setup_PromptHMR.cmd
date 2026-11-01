@echo off
setlocal DisableDelayedExpansion
cd /d "%~dp0"
echo PromptHMR optional Windows runtime setup. Close RGB MoCap first.
echo Separate Python 3.11, Torch 2.7.1 CUDA 12.8. Models are obtained separately.
echo Allow about 18 GiB free for setup, excluding models and compiler tools.
echo Native SAM2 build needs Visual Studio 2022 C++ Build Tools and CUDA Toolkit 12.8.
echo Review https://github.com/yufu-wang/PromptHMR/blob/main/LICENSE
echo Non-commercial scientific research use only; other dependencies have separate licenses.
choice /c YN /n /m "I reviewed these licenses and agree to the permitted use. Continue? [Y/N] "
if errorlevel 2 exit /b 1
if not exist "runtime\python.exe" (
  echo Run install.cmd first.
  pause
  exit /b 1
)
"runtime\python.exe" -I -X utf8 -B tools\setup_prompthmr.py --accept-licenses
set "phmr_result=%errorlevel%"
pause
exit /b %phmr_result%
