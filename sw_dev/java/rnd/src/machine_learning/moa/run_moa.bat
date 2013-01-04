@echo off
setlocal

rem export PATH=/home/sangwook/work_center/sw_dev/java/ext/bin:$PATH
set PATH=D:\work_center\sw_dev\java\ext\bin;%PATH%

rem export JAVA_HOME=/usr/lib/jvm/java-7-openjdk-amd64
rem set JAVA_HOME=C:\Program Files\Java\jre7
set JAVA_HOME=C:\Program Files\Java\jdk1.7.0_04

rem export MOA_HOME=/usr/local/lib/moa
set MOA_HOME=D:\work_center\sw_dev\java\rnd\src\machine_learning\moa\moa-release-2012.08.31

rem export LIB_PATH1=/usr/local/lib
set LIB_PATH1=D:\work_center\sw_dev\java\ext\lib
set LIB_PATH2=D:\work_center\sw_dev\java\rnd\lib

rem export CLASSPATH=.:$LIB_PATH2/moa.jar:$CLASSPATH
set CLASSPATH=.;%LIB_PATH2%\moa.jar;%CLASSPATH%

rem export MAX_JAVA_MEMORY=800
rem set MAX_JAVA_MEMORY=800

rem export HEAP_OPTION=-Xmx1000M
rem set HEAP_OPTION=-Xmx1000M

rem java $HEAP_OPTION -javaagent:$LIB_PATH2\sizeofag-1.0.0.jar moa.gui.GUI
java %HEAP_OPTION% -javaagent:%LIB_PATH2%\sizeofag-1.0.0.jar moa.gui.GUI

endlocal
echo on
