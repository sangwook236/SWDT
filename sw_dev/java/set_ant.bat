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

rem set JAVA_HOME=C:\Program Files\Java\jre1.8.0_40
set JAVA_HOME=C:\Program Files\Java\jdk1.8.0_40
set ANT_HOME=D:\util_portable\apache-ant-1.9.6\bin

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
