@echo off
setlocal

rem Usage -------------------------------------------------

rem -------------------------------------------------------

rem set JAVA_HOME=C:\Program Files\Java\jre7
set JAVA_HOME=C:\Program Files\Java\jdk1.7.0_04

set PATH=D:\work\sw_dev\java\ext\bin;%PATH%

rem -------------------------------------------------------

set LOCAL_LIB_PATH=D:\usr\local\lib
set JAVA_EXT_LIB_PATH=D:\work\sw_dev\java\ext\lib
set JAVA_RND_LIB_PATH=D:\work\sw_dev\java\rnd\lib

rem set CLASSPATH=.;%LOCAL_LIB_PATH%\<jar-file>;%CLASSPATH%

rem set MAX_JAVA_MEMORY=800
rem set HEAP_OPTION=-Xms4096m -Xmx8192m

rem -------------------------------------------------------

java %HEAP_OPTION% -jar %LOCAL_LIB_PATH%\<runnable-jar-file> <class-name>

rem Error -------------------------------------------------
rem java %HEAP_OPTION% -jar %LOCAL_LIB_PATH%\<non-runnable-jar-file>

endlocal
echo on
