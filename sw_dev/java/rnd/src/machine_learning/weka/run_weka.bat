@echo off
setlocal

rem export JAVA_HOME=/usr/lib/jvm/java-7-openjdk-amd64
rem set JAVA_HOME=C:\Program Files\Java\jre7
set JAVA_HOME=C:\Program Files\Java\jdk1.7.0_04

rem export WEKA_HOME=/work_center/sw_dev/java/\rnd/src/machine_learning/weka/weka-3-7-7
set WEKA_HOME=D:\work_center\sw_dev\java\rnd\src\machine_learning\weka\weka-3-7-7

rem export HEAP_OPTION=-Xmx1000M
set HEAP_OPTION=-Xmx1000M

rem java $HEAP_OPTION -jar $WEKA_HOME\weka.jar
java %HEAP_OPTION% -jar %WEKA_HOME%\weka.jar

endlocal
echo on
