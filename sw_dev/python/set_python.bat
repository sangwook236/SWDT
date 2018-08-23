@echo off
rem setlocal

if "%1%" == "--set" goto SET
if "%1%" == "--clean" goto CLEAN

echo.
echo Usage :
echo   "%0% [--set | --clean]"
echo.

goto EXIT

rem -----------------------------------------------------------
:SET

set PYTHONPATH=D:\util\Anaconda3

set SAVED_PATH_FOR_PYTHON=%PATH%
set PATH=%PYTHONPATH%\bin;%PATH%

goto EXIT

rem -----------------------------------------------------------
:CLEAN

set PYTHONPATH=

set PATH=%SAVED_PATH_FOR_PYTHON%
set SAVED_PATH_FOR_PYTHON=

rem -----------------------------------------------------------
:EXIT

rem endlocal
echo on
