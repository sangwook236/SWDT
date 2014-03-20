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
if "%ANT_HOME%" == "" (
	echo Error: can't find ant home
	goto EXIT
)

if "%AGLET_HOME%" == "" (
	set AGLET_HOME=E:\TestDir\Aglet\bin
	set AGLET_HOME_IS_DEFINED=true
)

if "%AGLET_CLASSPATH%" == "" (
	rem set CLASSPATH=%JAVA_HOME%\jre\lib;%JAVA_HOME%\lib
	set AGLET_CLASSPATH=%CLASSPATH%;%AGLET_HOME%\lib\aglets-2.0.2.jar;.
	set AGLET_CLASSPATH_IS_DEFINED=true
)

if "%AGLET_OLD_PATH%" == "" (
	set AGLET_OLD_PATH=%PATH%
	rem set PATH=%JAVA_HOME%\bin;%PATH%
	set PATH=%AGLET_HOME%\bin;%PATH%
	set AGLET_OLD_PATH_IS_DEFINED=true
)

goto EXIT

rem -----------------------------------------------------------
:CLEAN

if "%AGLET_HOME_IS_DEFINED%" == "true" (
	set AGLET_HOME=
	set AGLET_HOME_IS_DEFINED=
)

if "%AGLET_CLASSPATH_IS_DEFINED%" == "true" (
	set AGLET_CLASSPATH=
	set AGLET_CLASSPATH_IS_DEFINED=
)

if "%AGLET_OLD_PATH%" == "true" (
	set PATH=%AGLET_OLD_PATH%
	set AGLET_OLD_PATH=
	set AGLET_OLD_PATH_IS_DEFINED=
)

rem -----------------------------------------------------------
:EXIT

rem endlocal
echo on
