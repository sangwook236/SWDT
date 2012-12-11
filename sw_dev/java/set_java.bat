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

if "%JAVA_HOME%" == "" (
	rem set JAVA_HOME=C:\Program Files\Java\jre7
	set JAVA_HOME=C:\Program Files\Java\jdk1.7.0_04
	set JAVA_HOME_IS_DEFINED=true
)

if "%JAVA_OLD_PATH%" == "" (
	set JAVA_OLD_PATH=%PATH%
	set PATH=%JAVA_HOME%\bin;%PATH%
	set JAVA_OLD_PATH_IS_DEFINED=true
)

goto EXIT

rem -----------------------------------------------------------
:CLEAN

rem if not "%JAVA_HOME_IS_DEFINED%" == "true" (
if "%JAVA_HOME_IS_DEFINED%" == "true" (
	set JAVA_HOME=
	set JAVA_HOME_IS_DEFINED=
)

if "%JAVA_OLD_PATH_IS_DEFINED%" == "true" (
	set PATH=%JAVA_OLD_PATH%
	set JAVA_OLD_PATH=
	set JAVA_OLD_PATH_IS_DEFINED=
)

rem -----------------------------------------------------------
:EXIT

rem endlocal
echo on
