@echo off
setlocal

rem export PATH=/home/sangwook/work_center/sw_dev/java/ext/bin:$PATH
set PATH=D:\work_center\sw_dev\java\ext\bin;%PATH%

rem export JAVA_HOME=/usr/lib/jvm/java-7-openjdk-amd64
rem set JAVA_HOME=C:\Program Files\Java\jre7
set JAVA_HOME=C:\Program Files\Java\jdk1.7.0_04

rem export WEKA_HOME=/home/sangwook/work_center/sw_dev/java/rnd/src/machine_learning/weka/weka-3-7-7
set WEKA_HOME=D:\work_center\sw_dev\java\rnd\src\machine_learning\weka\weka-3-7-7

rem export R_HOME=/usr/local/lib/R
set R_HOME=D:\MyProgramFiles\R\R-2.15.0

rem export LIB_PATH1=/usr/local/lib
set LIB_PATH1=D:\work_center\sw_dev\java\ext\lib
set LIB_PATH2=D:\work_center\sw_dev\java\rnd\lib

rem export CLASSPATH=.:$LIB_PATH1/j3dcore.jar:$LIB_PATH1/j3dutils.jar:$LIB_PATH1/sqlite-jdbc-3.7.2.jar:$CLASSPATH
set CLASSPATH=.;%LIB_PATH1%\j3dcore.jar;%LIB_PATH1%\j3dutils.jar;%LIB_PATH1%\sqlite-jdbc-3.7.2.jar;%CLASSPATH%

rem export MAX_JAVA_MEMORY=800
rem set MAX_JAVA_MEMORY=800

rem export HEAP_OPTION=-Xmx1000M
set HEAP_OPTION=-Xmx1000M

rem java $HEAP_OPTION -jar $WEKA_HOME/weka.jar
java %HEAP_OPTION% -jar %WEKA_HOME%\weka.jar

endlocal
echo on
