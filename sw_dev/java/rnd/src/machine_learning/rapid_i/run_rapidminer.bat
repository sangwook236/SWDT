@echo off
setlocal

rem export JAVA_HOME=/usr/lib/jvm/java-7-openjdk-amd64
rem set JAVA_HOME=C:\Program Files\Java\jre7
set JAVA_HOME=C:\Program Files\Java\jdk1.7.0_04

rem export RAPIDMINER_HOME=/usr/lib/Rapid-I/RapidMiner5
rem export RAPIDMINER_HOME=/home/sangwook/work_center/sw_dev/java/rnd/src/machine_learning/rapid_i/rapidminer-5.2.008/rapidminer
rem set RAPIDMINER_HOME=D:\MyProgramFiles\Rapid-I\RapidMiner5
set RAPIDMINER_HOME=D:\work_center\sw_dev\java\rnd\src\machine_learning\rapid_i\rapidminer-5.2.008\rapidminer

rem export LIB_PATH1=/usr/local/lib
set LIB_PATH1=D:\work_center\sw_dev\java\ext\lib

rem export CLASSPATH=.:$LIB_PATH1/???.jar:$CLASSPATH
rem set CLASSPATH=.;%LIB_PATH1%\???.jar;%CLASSPATH%

rem export MAX_JAVA_MEMORY=800
set MAX_JAVA_MEMORY=800

rem export HEAP_OPTION=-Xms4096m -Xmx8192m
rem set HEAP_OPTION=-Xms4096m -Xmx8192m

rem java $HEAP_OPTION -Drapidminer.home=$RAPIDMINER_HOME -jar $RAPIDMINER_HOME/lib/rapidminer.jar
java %HEAP_OPTION% -Drapidminer.home=%RAPIDMINER_HOME% -jar %RAPIDMINER_HOME%\lib\rapidminer.jar

endlocal
echo on
