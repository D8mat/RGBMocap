@echo off
setlocal
set "ROOT=%~dp0..\.."
set "BUILD_PYTHON=%ROOT%\.setup\python\cpython-3.10.22-windows-x86_64-none\python.exe"
if not exist "%BUILD_PYTHON%" (
 echo Install the runtime first; the private build Python is missing.
 exit /b 1
)
cd /d "%ROOT%\third_party\KimodoMotionCorrection"
"%BUILD_PYTHON%" -I -X utf8 setup.py build_ext --build-temp "%ROOT%\.setup\motion-correction-build\temp" --build-lib "%ROOT%\.setup\motion-correction-build\lib"
exit /b %errorlevel%
