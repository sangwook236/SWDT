@echo off
rem setlocal

if "%1%" == "--set" goto SET
if "%1%" == "--clean" goto CLEAN

echo.
echo Usage :
echo   "%0% [--set | --clean]"
echo.
goto EXIT

rem -----------------------------------------------------------
:SET

set OLD_PATH_FOR_SPARK = %PATH%
set PATH=%SystemRoot%\system32;%PATH%

rem If winutils.exe is in D:/util_portable/spark-2.3.1-bin-hadoop2.7/bin:
set HADOOP_HOME=D:\util_portable\spark-2.3.1-bin-hadoop2.7

rem set JAVA_HOME="C:\Program Files\Java\jre1.8.0_181"
set JAVA_HOME=C:\Program Files\Java\jdk1.8.0_112
set MAVEN_HOME=D:\util_portable\apache-maven-3.5.4
set SBT_HOME=D:\util_portable\sbt-1.1.6\sbt
set SPARK_HOME=D:\util_portable\spark-2.3.1-bin-hadoop2.7

goto EXIT

rem -----------------------------------------------------------
:CLEAN

set SPARK_HOME=
set SBT_HOME=
set MAVEN_HOME=
set JAVA_HOME=

set HADOOP_HOME=

set PATH=%OLD_PATH_FOR_SPARK%

rem -----------------------------------------------------------
:EXIT

rem endlocal
echo on
