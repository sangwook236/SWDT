@echo off

if "%1%" == "--set" goto SET
if "%1%" == "--clean" goto CLEAN

echo.
echo Usage :
echo   "set_jade [--set | --clean]"
echo.
goto EXIT

rem -----------------------------------------------------------
:SET

if "%JAVA_HOME%" == "" (
	echo Error: can't find java home
	goto EXIT
)

if "%JADE_HOME%" == "" (
	set JADE_HOME=E:\TestDir\Jade\JADE-all-3.2\jade
)
if "%JADE_CLASSPATH%" == "" (
	rem set CLASSPATH=%JAVA_HOME%\jre\lib;%JAVA_HOME%\lib
	set JADE_CLASSPATH=%CLASSPATH%;%JADE_HOME%\lib\jade.jar;%JADE_HOME%\lib\jadeTools.jar;%JADE_HOME%\lib\Base64.jar;%JADE_HOME%\lib\iiop.jar;.
)
if "%JADE_OLD_PATH%" == "" (
	rem set JADE_OLD_PATH=%PATH%
	rem set PATH=%JAVA_HOME%\bin;%PATH%
)

goto EXIT

rem -----------------------------------------------------------
:CLEAN

set JADE_HOME=
set JADE_CLASSPATH=
if not "%JADE_OLD_PATH%" == "" (
	rem set PATH=%JADE_OLD_PATH%
	rem set JADE_OLD_PATH=
)

rem -----------------------------------------------------------
:EXIT

echo on

