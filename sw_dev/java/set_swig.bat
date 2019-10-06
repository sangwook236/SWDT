@echo off
rem setlocal

if "%1%" == "--set" goto SET
if "%1%" == "--clean" goto CLEAN

echo.
echo Usage:
echo   "%0% [--set | --clean]"
echo.

goto EXIT

rem -----------------------------------------------------------
:SET

set SWIG_HOME=D:\lib_repo\cpp\ext\swig_github
set PYTHON_HOME=D:\util\Anaconda3
set PYTHON_INCLUDE=%PYTHON_HOME%\include
set PYTHON_LIB=%PYTHON_HOME%\libs\python36.lib

set SAVED_PATH_FOR_SWIG=%PATH%
set PATH=%SWIG_HOME%\bin;%PATH%

goto EXIT

rem -----------------------------------------------------------
:CLEAN

set PYTHON_LIB=
set PYTHON_INCLUDE=
set PYTHON_HOME=
set SWIG_HOME=

set PATH=%SAVED_PATH_FOR_SWIG%
set SAVED_PATH_FOR_SWIG=

rem -----------------------------------------------------------
:EXIT

rem endlocal
echo on
