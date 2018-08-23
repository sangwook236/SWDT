@echo off
setlocal

set PATH=D:\util\R\R-3.2.3\bin\x64;%PATH%

R CMD BATCH %1%

endlocal
echo on
