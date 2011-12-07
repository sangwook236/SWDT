@echo off
setlocal

set BOOST_PREFIX_DIR="C:\boost"

set MSVC_ROOT="F:\Program Files\Microsoft Visual Studio\VC98"
set VISUALC="F:\Program Files\Microsoft Visual Studio\VC98"

set STLPORT_PATH="D:\WorkingDir\Development\ExternalLib\Cpp\src\stl\stlport"
set STLPORT_VERSION=4.6.2

set PYTHON_ROOT="F:\Program Files\Python24"
set PYTHON_VERSION=2.4
set PYTHON_LIB_PATH="F:\Program Files\Python24\Libs"

set TOOLS=msvc-stlport

bjam --prefix=%BOOST_PREFIX_DIR% "-sTOOLS=msvc-stlport" "-sBUILD=debug release <native-wchar_t>on <runtime-link>static/dynamic <threading>single/multi <stlport-iostream>on <stlport-cstd-namespace>std" %1%

rem set TOOLS=

rem set PYTHON_ROOT=
rem set PYTHON_VERSION=
rem set PYTHON_LIB_PATH=

rem set STLPORT_PATH=
rem set STLPORT_VERSION=

rem set MSVC_ROOT=
rem set VISUALC=

rem set BOOST_PREFIX_DIR=

endlocal
echo on
