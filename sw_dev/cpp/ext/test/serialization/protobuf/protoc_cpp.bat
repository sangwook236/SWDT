@echo off
setlocal

rem usage -------------------------------------------------
rem protoc_cpp.bat example.proto

rem -------------------------------------------------------

set PROTOC_HOME=D:\work_center\sw_dev\cpp\ext\src\serialization\protobuf\protoc-2.4.1-win32
set PATH=%PROTOC_HOME%;%PATH%

rem -------------------------------------------------------

protoc.exe --cpp_out=. %1%

endlocal
echo on
