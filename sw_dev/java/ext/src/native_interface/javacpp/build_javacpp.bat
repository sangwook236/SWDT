@echo off
setlocal

rem if 32-bit, "Visual Studio Command Prompt (2010)" must be used
rem if 64-bit, "Visual Studio x64 Win64 Command Prompt (2010)" must be used

rem build ...
javac -cp  .;..\..\..\lib\javacpp.jar %1%.java 
java  -jar ..\..\..\lib\javacpp.jar %1%

rem run ...
rem java  -cp  .;..\..\..\lib\javacpp.jar %1%

endlocal
echo on
