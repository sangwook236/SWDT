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

set JAVA_HOME=D:\util_portable\compiler\jdk-17_windows-x64_bin\jdk-17.0.7
rem set JAVA_HOME=D:\util_portable\compiler\jdk-20_windows-x64_bin\jdk-20.0.1

set SAVED_PATH_FOR_JAVA=%PATH%
set PATH=%JAVA_HOME%\bin;%PATH%

goto EXIT

rem -----------------------------------------------------------
:CLEAN

set JAVA_HOME=

set PATH=%SAVED_PATH_FOR_JAVA%
set SAVED_PATH_FOR_JAVA=

rem -----------------------------------------------------------
:EXIT

rem endlocal
echo on
