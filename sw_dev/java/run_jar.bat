@echo off
setlocal

rem export JAVA_HOME=/usr/lib/jvm/java-7-openjdk-amd64
rem set JAVA_HOME=C:\Program Files\Java\jre7
set JAVA_HOME=C:\Program Files\Java\jdk1.7.0_04

rem export LIB_PATH1=/usr/local/lib
set LIB_PATH1=D:\work_center\sw_dev\java

rem export CLASSPATH=.:$LIB_PATH1/runnable_jar.jar:$CLASSPATH
rem set CLASSPATH=.;%LIB_PATH1%\runnable_jar.jar;%CLASSPATH%

rem export MAX_JAVA_MEMORY=800
rem set MAX_JAVA_MEMORY=800

rem export HEAP_OPTION=-Xms4096m -Xmx8192m
rem set HEAP_OPTION=-Xms4096m -Xmx8192m

rem java %HEAP_OPTION% -classpath %LIB_PATH1%\nonrunnable_jar.jar java_jar.Hello
rem java %HEAP_OPTION% -classpath %LIB_PATH1%\nonrunnable_jar.jar java_jar.Hi
rem java %HEAP_OPTION% -classpath %LIB_PATH1%\runnable_jar.jar java_jar.Hello
rem java %HEAP_OPTION% -classpath %LIB_PATH1%\runnable_jar.jar java_jar.Hi

rem java $HEAP_OPTION -jar $LIB_PATH1/runnable_jar.jar java_jar.Hello
rem java $HEAP_OPTION -jar $LIB_PATH1/runnable_jar.jar java_jar.Hi
rem java %HEAP_OPTION% -jar %LIB_PATH1%\runnable_jar.jar java_jar.Hello
java %HEAP_OPTION% -jar %LIB_PATH1%\runnable_jar.jar java_jar.Hi

rem << error >>
rem java $HEAP_OPTION -jar $LIB_PATH/nonrunnable_jar.jar
rem java %HEAP_OPTION% -jar %LIB_PATH%\nonrunnable_jar.jar

endlocal
echo on
