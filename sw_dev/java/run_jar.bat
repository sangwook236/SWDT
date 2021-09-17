@echo off
setlocal

rem Usage -------------------------------------------------

rem -------------------------------------------------------

set JAVA_HOME=D:\util_portable\jdk-17_windows-x64_bin\jdk-17

set PATH=D:\work\SWDT_github\sw_dev\java\ext\bin;%PATH%

rem -------------------------------------------------------

set LOCAL_LIB_PATH=D:\usr\local\lib
set JAVA_EXT_LIB_PATH=D:\work\SWDT_github\sw_dev\java\ext\lib
set JAVA_RND_LIB_PATH=D:\work\SWDT_github\sw_dev\java\rnd\lib

rem set CLASSPATH=.;%LOCAL_LIB_PATH%\<jar-file>;%CLASSPATH%

rem set MAX_JAVA_MEMORY=800
rem set HEAP_OPTION=-Xms4096m -Xmx8192m

rem -------------------------------------------------------

call java %HEAP_OPTION% -jar %LOCAL_LIB_PATH%\<runnable-jar-file> <class-name>
rem start /b java %HEAP_OPTION% -jar %LOCAL_LIB_PATH%\<runnable-jar-file> <class-name>
rem start /min java %HEAP_OPTION% -jar %LOCAL_LIB_PATH%\<runnable-jar-file> <class-name>

rem Error -------------------------------------------------
rem call java %HEAP_OPTION% -jar %LOCAL_LIB_PATH%\<non-runnable-jar-file>

endlocal
echo on
