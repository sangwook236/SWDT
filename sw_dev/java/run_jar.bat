@echo off
setlocal

rem export LIB_PATH=lib
set LIB_PATH=D:\work_center\sw_dev\java\ext\lib

rem export CLASSPATH=.:$LIB_PATH/runnable_jar.jar:$CLASSPATH
rem set CLASSPATH=.;%LIB_PATH%\runnable_jar.jar;%CLASSPATH%

rem export HEAP_OPTION=-Xms4096m -Xmx8192m
rem set HEAP_OPTION=-Xms4096m -Xmx8192m

rem error
rem java $HEAP_OPTION -jar $LIB_PATH/nonrunnable_jar.jar
rem java %HEAP_OPTION% -jar %LIB_PATH%\nonrunnable_jar.jar

rem java $HEAP_OPTION -jar $LIB_PATH/runnable_jar.jar
java %HEAP_OPTION% -jar %LIB_PATH%\runnable_jar.jar

java %HEAP_OPTION% -classpath %LIB_PATH%\nonrunnable_jar.jar java_jar.Hello
java %HEAP_OPTION% -classpath %LIB_PATH%\nonrunnable_jar.jar java_jar.Hi
java %HEAP_OPTION% -classpath %LIB_PATH%\runnable_jar.jar java_jar.Hello
java %HEAP_OPTION% -classpath %LIB_PATH%\runnable_jar.jar java_jar.Hi

endlocal
echo on
