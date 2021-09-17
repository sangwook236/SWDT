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
set MAVEN_HOME=D:\util_portable\build_tool\apache-maven-3.5.4

set SAVED_PATH_FOR_MAVEN=%PATH%
set PATH=%MAVEN_HOME%\bin;%JAVA_HOME%\bin;%PATH%

goto EXIT

rem -----------------------------------------------------------
:CLEAN

set MAVEN_HOME=
set JAVA_HOME=

set PATH=%SAVED_PATH_FOR_MAVEN%
set SAVED_PATH_FOR_MAVEN=

rem -----------------------------------------------------------
:EXIT

rem endlocal
echo on
