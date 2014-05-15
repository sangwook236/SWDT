@echo off
setlocal

set path=D:\MyProgramFiles\R\R-3.0.3\bin\x64;%path%

R CMD BATCH %1%

endlocal
echo on
