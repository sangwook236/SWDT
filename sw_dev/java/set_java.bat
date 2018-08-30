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
