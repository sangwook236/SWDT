@echo off
setlocal

rem set JAVA_HOME=C:\Program Files\Java\jre7
set JAVA_HOME=C:\Program Files\Java\jdk1.7.0_04

set PATH=D:\work_center\sw_dev\java\ext\bin;%PATH%

rem -------------------------------------------------------

set LIB_PATH1=D:\work_center\sw_dev\java\ext\lib
set LIB_PATH2=D:\work_center\sw_dev\java\rnd\lib

rem set CLASSPATH=.;%LIB_PATH1%\runnable_jar.jar;%CLASSPATH%

rem set MAX_JAVA_MEMORY=800
rem set HEAP_OPTION=-Xms4096m -Xmx8192m

rem -------------------------------------------------------

rem java %HEAP_OPTION% -jar %LIB_PATH1%\runnable_jar.jar java_jar.Hello
java %HEAP_OPTION% -jar %LIB_PATH1%\runnable_jar.jar java_jar.Hi

rem error -------------------------------------------------
rem java %HEAP_OPTION% -jar %LIB_PATH%\nonrunnable_jar.jar

endlocal
echo on
