@echo off
setlocal

rem [e.g.] run_pivot_script hello.bxml

rem export LIB_PATH=lib
set LIB_PATH=D:\work_center\sw_dev\java\ext\lib

rem export CLASSPATH=.:$LIB_PATH/pivot-core-2.0.2.jar:$LIB_PATH/pivot-wtk-2.0.2.jar:$LIB_PATH/pivot-wtk-terra-2.0.2.jar:$CLASSPATH
set CLASSPATH=.;%LIB_PATH%\pivot-core-2.0.2.jar;%LIB_PATH%\pivot-wtk-2.0.2.jar;%LIB_PATH%\pivot-wtk-terra-2.0.2.jar;%CLASSPATH%

rem export HEAP_OPTION=-Xms4096m -Xmx8192m
rem set HEAP_OPTION=-Xms4096m -Xmx8192m

rem java $HEAP_OPTION org.apache.pivot.wtk.ScriptApplication --src=$@
java %HEAP_OPTION% org.apache.pivot.wtk.ScriptApplication --src=/%1%

endlocal
echo on
