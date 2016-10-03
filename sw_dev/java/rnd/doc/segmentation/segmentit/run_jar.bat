@echo off
setlocal

rem usage -------------------------------------------------

rem -------------------------------------------------------

rem set JAVA_HOME=C:\Program Files\Java\jre7
set JAVA_HOME=C:\Program Files\Java\jdk1.7.0_04

set PATH=D:\work\sw_dev\java\ext\bin;%PATH%

rem -------------------------------------------------------

set LOCAL_LIB_PATH=D:\usr\local\lib
set JAVA_EXT_LIB_PATH=D:\work\sw_dev\java\ext\lib
set JAVA_RND_LIB_PATH=D:\work\sw_dev\java\rnd\lib

rem set CLASSPATH=.;%LOCAL_LIB_PATH%\SegmentIt_1.0.3.jar;%CLASSPATH%

rem set MAX_JAVA_MEMORY=800
rem set HEAP_OPTION=-Xms4096m -Xmx8192m

rem -------------------------------------------------------

java %HEAP_OPTION% -jar SegmentIt_1.0.3.jar

endlocal
echo on
