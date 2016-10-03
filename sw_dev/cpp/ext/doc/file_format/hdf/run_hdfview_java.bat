@echo off
setlocal

rem usage -------------------------------------------------

rem -------------------------------------------------------

rem set JAVAHOME=C:\Program Files\Java\jre7
set JAVAHOME=C:\Program Files\Java\jdk1.7.0_04

set HDFVIEW_HOME=D:\work_center\sw_dev\cpp\ext\src\file_format\hdf\hdf-java-2.9-bin\hdf-java
set HDFVIEW_INSTALL=%HDFVIEW_HOME%\lib

set PATH=%HDFVIEW_JAVA_HOME%\bin;%JAVAHOME%\bin;%PATH%

rem -------------------------------------------------------

set CLASSPATH=%HDFVIEW_INSTALL%\*;%HDFVIEW_INSTALL%\ext\*
set LIB_PATH=%HDFVIEW_INSTALL%;%HDFVIEW_INSTALL%\ext

set HEAP_OPTION=-Xmx1024m

rem -------------------------------------------------------

java %HEAP_OPTION% -Djava.library.path=%LIB_PATH% -Dhdfview.root=%HDFVIEW_INSTALL% ncsa.hdf.view.HDFView -root %HDFVIEW_INSTALL%

endlocal
echo on
