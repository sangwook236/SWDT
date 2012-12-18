@echo off
setlocal

rem export JAVA_HOME=/usr/lib/jvm/java-7-openjdk-amd64
rem set JAVA_HOME=C:\Program Files\Java\jre7
set JAVA_HOME=C:\Program Files\Java\jdk1.7.0_04

rem export LIB_PATH1=lib
set LIB_PATH1=D:\work_center\sw_dev\java\ext\lib

rem export CLASSPATH=.:$LIB_PATH1/runnable_jar.jar:$CLASSPATH
rem set CLASSPATH=.;%LIB_PATH1%\runnable_jar.jar;%CLASSPATH%

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
