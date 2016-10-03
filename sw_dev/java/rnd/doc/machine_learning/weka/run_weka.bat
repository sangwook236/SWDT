@echo off
setlocal

rem usage -------------------------------------------------

rem -------------------------------------------------------

rem set JAVA_HOME=C:\Program Files\Java\jre7
set JAVA_HOME=C:\Program Files\Java\jdk1.7.0_04
set WEKA_HOME=D:\work_center\sw_dev\java\rnd\src\machine_learning\weka\weka-3-7-7
set R_HOME=D:\MyProgramFiles\R\R-2.15.0

set PATH=D:\work_center\sw_dev\java\ext\bin;%PATH%

rem -------------------------------------------------------

set LIB_PATH1=D:\work_center\sw_dev\java\ext\lib
set LIB_PATH2=D:\work_center\sw_dev\java\rnd\lib

set CLASSPATH=.;%LIB_PATH1%\sqlite-jdbc-3.7.2.jar;%LIB_PATH1%\mysql-connector-java-5.1.22-bin.jar;%LIB_PATH1%\j3dcore.jar;%LIB_PATH1%\j3dutils.jar;%CLASSPATH%

rem set MAX_JAVA_MEMORY=800
set HEAP_OPTION=-Xmx1000M

rem -------------------------------------------------------

java %HEAP_OPTION% -jar %WEKA_HOME%\weka.jar

endlocal
echo on
