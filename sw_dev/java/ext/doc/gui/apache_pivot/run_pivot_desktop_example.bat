@echo off
setlocal

rem [e.g.] run_pivot_desktop_example

rem export JAVA_HOME=/usr/lib/jvm/java-7-openjdk-amd64
rem set JAVA_HOME=C:\Program Files\Java\jre7
set JAVA_HOME=C:\Program Files\Java\jdk1.7.0_04

rem export LIB_PATH=lib
set LIB_PATH=D:\work_center\sw_dev\java\ext\src\gui_framework\apache_pivot\apache-pivot-2.0.2-src\lib

rem export CLASSPATH=.:$LIB_PATH/pivot-core-2.0.2.jar:$LIB_PATH/pivot-wtk-2.0.2.jar:$LIB_PATH/pivot-wtk-terra-2.0.2.jar:$LIB_PATH/pivot-tutorials-2.0.2.jar:$CLASSPATH
set CLASSPATH=.;%LIB_PATH%\pivot-core-2.0.2.jar;%LIB_PATH%\pivot-wtk-2.0.2.jar;%LIB_PATH%\pivot-wtk-terra-2.0.2.jar;%LIB_PATH%\pivot-tutorials-2.0.2.jar;%CLASSPATH%

rem export HEAP_OPTION=-Xms4096m -Xmx8192m
rem set HEAP_OPTION=-Xms4096m -Xmx8192m

rem java $HEAP_OPTION org.apache.pivot.wtk.DesktopApplicationContext org.apache.pivot.tutorials.HelloBXML
java %HEAP_OPTION% org.apache.pivot.wtk.DesktopApplicationContext org.apache.pivot.tutorials.HelloBXML

endlocal
echo on
