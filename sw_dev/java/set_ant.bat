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
	set ANT_HOME=D:\util\software_build\ant\apache-ant-1.8.4-bin\apache-ant-1.8.4\bin
	set ANT_HOME_IS_DEFINED=true
)

if "%ANT_OLD_PATH%" == "" (
	set ANT_OLD_PATH=%PATH%
	set PATH=%ANT_HOME%\bin;%PATH%
	set ANT_OLD_PATH_IS_DEFINED=true
)

goto EXIT

rem -----------------------------------------------------------
:CLEAN

if "%ANT_HOME_IS_DEFINED%" == "true" (
	set ANT_HOME=
	set ANT_HOME_IS_DEFINED=
)

if "%ANT_OLD_PATH_IS_DEFINED%" == "true" (
	set PATH=%ANT_OLD_PATH%
	set ANT_OLD_PATH=
	set ANT_OLD_PATH_IS_DEFINED=
)

rem -----------------------------------------------------------
:EXIT

rem endlocal
echo on
