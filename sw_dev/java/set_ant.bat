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

rem set JAVA_HOME=C:\Progra~1\Java\jre1.8.0_181
set JAVA_HOME=C:\Progra~1\Java\jdk1.8.0_112
set ANT_HOME=D:\util_portable\apache-ant-1.10.5

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
