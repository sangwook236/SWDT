@echo off
setlocal

rem if 32-bit, "Visual Studio Command Prompt (2010)" must be used
rem if 64-bit, "Visual Studio x64 Win64 Command Prompt (2010)" must be used

rem export JAVA_HOME=/usr/lib/jvm/java-7-openjdk-amd64
rem set JAVA_HOME=C:\Program Files\Java\jre7
set JAVA_HOME=C:\Program Files\Java\jdk1.7.0_04

rem export LIB_PATH1=lib
set LIB_PATH1=D:\work_center\sw_dev\java\ext\lib

rem export CLASSPATH=.:../../../lib/javacpp.jar:$CLASSPATH
rem set CLASSPATH=.;..\..\..\lib\javacpp.jar;%CLASSPATH%

rem export HEAP_OPTION=-Xms4096m -Xmx8192m
rem set HEAP_OPTION=-Xms4096m -Xmx8192m

rem build ...
rem javac -cp .:../../../lib/javacpp.jar $@.java 
javac -cp .;..\..\..\lib\javacpp.jar %1%.java 
rem java $HEAP_OPTION -jar ../../../lib/javacpp.jar $@
java %HEAP_OPTION% -jar ..\..\..\lib\javacpp.jar %1%

rem run ...
rem java $HEAP_OPTION -cp .:../../../lib/javacpp.jar $@
rem java %HEAP_OPTION% -cp .;..\..\..\lib\javacpp.jar %1%

endlocal
echo on
