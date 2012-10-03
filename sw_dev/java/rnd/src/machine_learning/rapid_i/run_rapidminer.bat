@echo off
setlocal

set JAVA_HOME=C:\Program Files\Java\jre7
set MAX_JAVA_MEMORY=800
set RAPIDMINER_HOME=D:\work_center\sw_dev\java\rnd\src\machine_learning\rapid_i\rapidminer-5.2.008\rapidminer

java -Drapidminer.home=D:\MyProgramFiles\RapidMiner-5.2.008 -jar %RAPIDMINER_HOME%\lib\rapidminer.jar

endlocal
echo on
