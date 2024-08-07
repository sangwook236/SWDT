@echo off
setlocal

rem usage -------------------------------------------------

rem -------------------------------------------------------

rem set JAVA_HOME=C:\Program Files\Java\jre7
set JAVA_HOME=C:\Program Files\Java\jdk1.7.0_04
rem set R_HOME=D:\MyProgramFiles\R\R-3.0.3
set R_HOME=D:\MyProgramFiles\R\R-3.0.3
rem set RAPIDMINER_HOME=D:\util\RapidMiner5
set RAPIDMINER_HOME=D:\util\rapidminer-5.2.008\rapidminer

set PATH=D:\work\SWDT_github\sw_dev\java\ext\bin;%PATH%

rem -------------------------------------------------------

set LIB_PATH1=D:\work\SWDT_github\sw_dev\java\ext\lib
set LIB_PATH2=D:\work\SWDT_github\sw_dev\java\rnd\lib

rem set CLASSPATH=.;%LIB_PATH1%\???.jar;%CLASSPATH%

set MAX_JAVA_MEMORY=800
rem set HEAP_OPTION=-Xms4096m -Xmx8192m

rem -------------------------------------------------------

java %HEAP_OPTION% -Drapidminer.home=%RAPIDMINER_HOME% -jar %RAPIDMINER_HOME%\lib\rapidminer.jar

endlocal
echo on
