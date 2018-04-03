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
	echo Error: can't find java home
	goto EXIT
)

if "%MAVEN_HOME%" == "" (
	set MAVEN_HOME=D:\util_portable\apache-maven-3.3.9
	set MAVEN_HOME_IS_DEFINED=true
)

if "%MAVEN_OLD_PATH%" == "" (
	set MAVEN_OLD_PATH=%PATH%
	set PATH=%MAVEN_HOME%\bin;%PATH%
	set MAVEN_OLD_PATH_IS_DEFINED=true
)

goto EXIT

rem -----------------------------------------------------------
:CLEAN

if "%MAVEN_HOME_IS_DEFINED%" == "true" (
	set MAVEN_HOME=
	set MAVEN_HOME_IS_DEFINED=
)

if "%MAVEN_OLD_PATH_IS_DEFINED%" == "true" (
	set PATH=%MAVEN_OLD_PATH%
	set MAVEN_OLD_PATH=
	set MAVEN_OLD_PATH_IS_DEFINED=
)

rem -----------------------------------------------------------
:EXIT

rem endlocal
echo on
