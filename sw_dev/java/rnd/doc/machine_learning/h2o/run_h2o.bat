@echo off
setlocal

rem usage -------------------------------------------------

rem -------------------------------------------------------

rem set JAVA_HOME=C:\Program Files\Java\jre7
set JAVA_HOME=C:\Program Files\Java\jdk1.7.0_04
set H2O_HOME=D:\util_portable\h2o-3.18.0.5

set PATH=D:\work\SWDT_github\sw_dev\java\ext\bin;%PATH%

rem -------------------------------------------------------

set CLASSPATH=.;%CLASSPATH%

rem -------------------------------------------------------

java -jar %H2O_HOME%\h2o.jar

endlocal
echo on
