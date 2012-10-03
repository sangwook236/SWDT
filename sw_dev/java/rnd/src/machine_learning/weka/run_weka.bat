@echo off
setlocal

set JAVA_HOME=C:\Program Files\Java\jre7
rem set WEKA_HOME=D:\work_center\sw_dev\java\rnd\src\machine_learning\weka\weka-3-6-8
set WEKA_HOME=D:\work_center\sw_dev\java\rnd\src\machine_learning\weka\weka-3-7-7

java -Xmx1000M -jar %WEKA_HOME%\weka.jar

endlocal
echo on
