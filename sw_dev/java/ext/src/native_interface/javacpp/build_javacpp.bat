@echo off
setlocal

rem if 32-bit, "Visual Studio Command Prompt (2010)" must be used
rem if 64-bit, "Visual Studio x64 Win64 Command Prompt (2010)" must be used

rem set JAVA_HOME=C:\Program Files\Java\jre7
set JAVA_HOME=C:\Program Files\Java\jdk1.7.0_04

set PATH=%JAVA_HOME%\bin;D:\work_center\sw_dev\java\ext\bin;%PATH%

rem -------------------------------------------------------

set LIB_PATH1=D:\work_center\sw_dev\java\ext\lib

rem set CLASSPATH=.;%LIB_PATH1%\javacpp.jar;%CLASSPATH%

rem set HEAP_OPTION=-Xms4096m -Xmx8192m

rem build -------------------------------------------------

javac -cp .;%LIB_PATH1%\javacpp.jar %1%.java
java %HEAP_OPTION% -jar %LIB_PATH1%\javacpp.jar %1%

rem run ---------------------------------------------------

rem java %HEAP_OPTION% -cp .;%LIB_PATH1%\javacpp.jar %1%

endlocal
echo on
