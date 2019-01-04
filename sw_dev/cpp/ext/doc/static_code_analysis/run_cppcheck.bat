@echo off
setlocal

set CPPCHECK_HOME=D:/lib_repo/cpp/ext/cppcheck-1.86
set PATH=%CPPCHECK_HOME%/bin;%PATH%
set CFGDIR=%CPPCHECK_HOME%/bin/cfg

cppcheck.exe %1%

endlocal
echo on
