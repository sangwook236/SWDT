@echo off
setlocal

set BOOST_PREFIX_DIR="C:\boost"

set VC71_ROOT="F:\Program Files\Microsoft Visual Studio .NET 2003\Vc7"

set STLPORT_PATH="D:\WorkingDir\Development\ExternalLib\Cpp\src\stl\stlport"
set STLPORT_VERSION=4.6.2

set PYTHON_ROOT="F:\Program Files\Python24"
set PYTHON_VERSION=2.4
set PYTHON_LIB_PATH="F:\Program Files\Python24\Libs"

set TOOLS=vc-7_1-stlport

rem bjam --prefix=%BOOST_PREFIX_DIR% "-sTOOLS=vc-7_1-stlport" "-sBUILD=debug release <runtime-link>static/dynamic <threading>single/multi <stlport-iostream>on <stlport-cstd-namespace>std" %1%
bjam --prefix=%BOOST_PREFIX_DIR% "-sTOOLS=vc-7_1-stlport" "-sBUILD=debug release <native-wchar_t>on <runtime-link>static/dynamic <threading>single/multi <stlport-iostream>on <stlport-cstd-namespace>std" %1%

rem set TOOLS=

rem set PYTHON_ROOT=
rem set PYTHON_VERSION=
rem set PYTHON_LIB_PATH=

rem set STLPORT_PATH=
rem set STLPORT_VERSION=

rem set VC71_ROOT=

rem set BOOST_PREFIX_DIR=

endlocal
echo on
