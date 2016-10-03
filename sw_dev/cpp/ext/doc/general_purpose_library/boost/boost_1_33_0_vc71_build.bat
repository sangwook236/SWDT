@echo off
setlocal

set BOOST_PREFIX_DIR="C:\boost"

set VC71_ROOT="F:\Program Files\Microsoft Visual Studio .NET 2003\Vc7"

rem set STLPORT_PATH="D:\WorkingDir\Development\ExternalLib\Cpp\src\stl\stlport\STLport-5.0RC6"
set STLPORT_PATH="D:\WorkingDir\Development\ExternalLib\Cpp\src\stl\stlport"
set STLPORT_VERSION=5.0RC6

set PYTHON_ROOT="F:\Program Files\Python24"
set PYTHON_VERSION=2.4
set PYTHON_LIB_PATH="F:\Program Files\Python24\Libs"

set BZIP2_BINARY=libbz2
set BZIP2_INCLUDE="D:\WorkingDir\Development\ExternalLib\Cpp\inc\bzip"
set BZIP2_LIBPATH="D:\WorkingDir\Development\ExternalLib\Cpp\src\bzip\bzip2-1.0.3\Release"
set BZIP2_SOURCE="D:\WorkingDir\Development\ExternalLib\Cpp\src\bzip\bzip2-1.0.3"

set ZLIB_BINARY=zdll
set ZLIB_INCLUDE="D:\WorkingDir\Development\ExternalLib\Cpp\inc\zlib"
set ZLIB_LIBPATH="D:\WorkingDir\Development\ExternalLib\Cpp\src\zlib\zlib123-dll\lib"
set ZLIB_SOURCE=

set ICU_PATH="D:\WorkingDir\Development\ExternalLib\Cpp\src\icu\icu-3.4"

set TOOLS=vc-7_1-stlport

rem bjam --prefix=%BOOST_PREFIX_DIR% "-sTOOLS=vc-7_1-stlport" "-sBUILD=debug release <runtime-link>static/dynamic <threading>single/multi <stlport-iostream>on <stlport-cstd-namespace>std" %1%
bjam --prefix=%BOOST_PREFIX_DIR% "-sTOOLS=vc-7_1-stlport" "-sBOOST_REGEX_DYN_LINK=1" "-sBUILD=debug release <runtime-link>static/dynamic <threading>single/multi <native-wchar_t>on <stlport-iostream>on <stlport-cstd-namespace>std" %1%

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
