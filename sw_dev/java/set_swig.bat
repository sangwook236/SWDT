@echo off
rem setlocal

if "%1%" == "--set" goto SET
if "%1%" == "--clean" goto CLEAN

echo.
echo Usage:
echo   "%0% [--set | --clean]"
echo.
goto EXIT

rem -----------------------------------------------------------
:SET

if "%SWIG_HOME%" == "" (
	set SWIG_HOME=D:\WorkingDir\Development\ExternalLib\Cpp\src\swig\SWIG-1.3.24
	set SWIG_HOME_IS_DEFINED=true
)

if "%SWIG_OLD_PATH%" == "" (
	set SWIG_OLD_PATH=%PATH%
	set PATH=%SWIG_HOME%;%PATH%
	set SWIG_OLD_PATH_IS_DEFINED=true
)

if "%PYTHON_HOME%" == "" (
	set PYTHON_HOME=F:\Program Files\Python24
	set PYTHON_HOME_IS_DEFINED=true
)

if "%PYTHON_INCLUDE%" == "" (
	set PYTHON_INCLUDE=%PYTHON_HOME%\include
	set PYTHON_INCLUDE_IS_DEFINED=true
)

if "%PYTHON_LIB%" == "" (
	set PYTHON_LIB=%PYTHON_HOME%\libs\python24.lib
	set PYTHON_LIB_IS_DEFINED=true
)

goto EXIT

rem -----------------------------------------------------------
:CLEAN

if "%PYTHON_HOME_IS_DEFINED%" == "true" (
	set PYTHON_HOME=
	set PYTHON_HOME_IS_DEFINED=
)
if "%PYTHON_INCLUDE_IS_DEFINED%" == "true" (
	set PYTHON_INCLUDE=
	set PYTHON_INCLUDE_IS_DEFINED=
)

if "%PYTHON_LIB_IS_DEFINED%" == "true" (
	set PYTHON_LIB=
	set PYTHON_LIB_IS_DEFINED=
)

if "%SWIG_HOME_IS_DEFINED%" == "true" (
	set SWIG_HOME=
	set SWIG_HOME_IS_DEFINED=
)

if "%SWIG_OLD_PATH_IS_DEFINED%" == "true" (
	set PATH=%SWIG_OLD_PATH%
	set SWIG_OLD_PATH=
	set SWIG_OLD_PATH_IS_DEFINED=
)

rem -----------------------------------------------------------
:EXIT

rem endlocal
echo on
