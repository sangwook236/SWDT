@echo off
setlocal

rem usage -------------------------------------------------

rem -------------------------------------------------------

set JAVA_HOME=C:\Program Files\Java\jre7
set HADOOP_HOME=D:\work_center\sw_dev\java\ext\src\hadoop\hadoop-0.23.4

set PATH=%PATH%;%HADOOP_HOME%\bin

rem -------------------------------------------------------

set HADOOP_CLASSPATH=.

rem -------------------------------------------------------

javac -cp .;%HADOOP_HOME%\share\hadoop\common\hadoop-common-0.23.4.jar;%HADOOP_HOME%\share\hadoop\mapreduce\hadoop-mapreduce-client-core-0.23.4.jar *.java

rem run ---------------------------------------------------

hadoop HadoopMain input\input_file.txt output

endlocal
echo on
