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
set MAVEN_HOME=D:\util_portable\apache-maven-3.3.9

goto EXIT

rem -----------------------------------------------------------
:CLEAN

set MAVEN_HOME=
set JAVA_HOME=

rem -----------------------------------------------------------
:EXIT

rem endlocal
echo on
