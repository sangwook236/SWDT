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

rem If winutils.exe is in D:/lib_repo/java/ext/winutils_github/hadoop-2.7.1/bin:
rem set HADOOP_HOME=D:\util_portable\hadoop-3.0.3
set HADOOP_HOME=D:\lib_repo\java\ext\winutils_github\hadoop-2.7.1

set JAVA_HOME=D:\util_portable\jdk-17_windows-x64_bin\jdk-17
set MAVEN_HOME=D:\util_portable\build_tool\apache-maven-3.5.4
set ANT_HOME=D:\util_portable\build_tool\apache-ant-1.10.5
set SBT_HOME=D:\util_portable\build_tool\sbt-1.1.6\sbt
set SPARK_HOME=D:\util_portable\spark-2.3.1-bin-hadoop2.7

set SAVED_PATH_FOR_SPARK=%PATH%
rem set PATH=%SPARK_HOME%\bin;%SBT_HOME%\bin;%ANT_HOME%\bin;%MAVEN_HOME%\bin;%JAVA_HOME%\bin;%SystemRoot%\system32
set PATH=%SPARK_HOME%\bin;%HADOOP_HOME%\bin;%SBT_HOME%\bin;%ANT_HOME%\bin;%MAVEN_HOME%\bin;%JAVA_HOME%\bin;%PATH%

goto EXIT

rem -----------------------------------------------------------
:CLEAN

set SPARK_HOME=
set SBT_HOME=
set MAVEN_HOME=
set JAVA_HOME=

set HADOOP_HOME=

set PATH=%SAVED_PATH_FOR_SPARK%
set SAVED_PATH_FOR_SPARK=

rem -----------------------------------------------------------
:EXIT

rem endlocal
echo on
