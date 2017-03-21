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

if "%PYTHONPATH%" == "" (
	set PYTHONPATH=D:\MyProgramFiles\Python35
	set PYTHONPATH_IS_DEFINED=true
)

if "%PYTHON_OLD_PATH%" == "" (
	set PYTHON_OLD_PATH=%PATH%
	set PATH=%PYTHONPATH%;%PYTHONPATH%\Scripts;%APPDATA%\Python\Python35\Scripts;%PATH%
	set PYTHON_OLD_PATH_IS_DEFINED=true
)

goto EXIT

rem -----------------------------------------------------------
:CLEAN

rem if not "%PYTHONPATH_IS_DEFINED%" == "true" (
if "%PYTHONPATH_IS_DEFINED%" == "true" (
	set PYTHONPATH=
	set PYTHONPATH_IS_DEFINED=
)

if "%PYTHON_OLD_PATH_IS_DEFINED%" == "true" (
	set PATH=%PYTHON_OLD_PATH%
	set PYTHON_OLD_PATH=
	set PYTHON_OLD_PATH_IS_DEFINED=
)

rem -----------------------------------------------------------
:EXIT

rem endlocal
echo on
