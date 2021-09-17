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

set JAVA_HOME=D:\util_portable\jdk-17_windows-x64_bin\jdk-17
set ANT_HOME=D:\util_portable\build_tool\apache-ant-1.10.5

set SAVED_PATH_FOR_ANT=%PATH%
set PATH=%ANT_HOME%\bin;%JAVA_HOME%\bin;%PATH%

goto EXIT

rem -----------------------------------------------------------
:CLEAN

set ANT_HOME=
set JAVA_HOME=

set PATH=%SAVED_PATH_FOR_ANT%
set SAVED_PATH_FOR_ANT=

rem -----------------------------------------------------------
:EXIT

rem endlocal
echo on
