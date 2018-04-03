@echo off
setlocal

rem usage -------------------------------------------------

rem -------------------------------------------------------

rem set JAVA_HOME=C:\Program Files\Java\jre7
set JAVA_HOME=C:\Program Files\Java\jdk1.7.0_04
set MOA_HOME=D:\util_portable\moa-release-2012.08.31

set PATH=D:\work\SWDT_github\sw_dev\java\ext\bin;%PATH%

rem -------------------------------------------------------

set LIB_PATH1=D:\work\SWDT_github\sw_dev\java\ext\lib
set LIB_PATH2=D:\work\SWDT_github\sw_dev\java\rnd\lib

set CLASSPATH=.;%LIB_PATH2%\moa.jar;%CLASSPATH%

rem set MAX_JAVA_MEMORY=800
rem set HEAP_OPTION=-Xmx1000M

rem -------------------------------------------------------

java %HEAP_OPTION% -javaagent:%LIB_PATH2%\sizeofag-1.0.0.jar moa.gui.GUI

endlocal
echo on
