@echo off
setlocal

set path=D:\MyProgramFiles\R\R-2.15.0\bin\x64;%path%

R CMD BATCH %1%

endlocal
echo on
