@echo off
setlocal

rem if 32-bit, "Visual Studio Command Prompt (2010)" must be used
rem if 64-bit, "Visual Studio x64 Win64 Command Prompt (2010)" must be used

rem build ...
rem javac -cp  .;..\..\..\lib\javacpp.jar %1%.java 
rem java  -jar ..\..\..\lib\javacpp.jar %1%

rem run ...
java  -cp  .;..\..\..\lib\javacpp.jar %1%

endlocal
echo on
