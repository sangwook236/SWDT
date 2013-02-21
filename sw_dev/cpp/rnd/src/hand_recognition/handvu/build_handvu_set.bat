@echo off
rem setlocal

rem usage -------------------------------------------------

rem -------------------------------------------------------

set OPENCV_BASE_INC_PATH=D:\work_center\sw_dev\cpp\rnd\inc\opencv-2.4.3
set OPENCV_INC_PATH=%OPENCV_BASE_INC_PATH%\opencv
set OPENCV_LIB_PATH=D:\work_center\sw_dev\cpp\rnd\lib

set INC_OPENCV=%OPENCV2_INC_PATH%
set INC_OPENCV_CXCORE=%OPENCV_BASE_INC_PATH%
set INC_OPENCV_AUX=%OPENCV_INC_PATH%
set INC_OPENCV_HIGHGUI=%OPENCV_INC_PATH%
set LIB_OPENCV=%OPENCV_LIB_PATH%

rem DirectShow Baseclasses path----------------------------
set INC_DX_BASECLASSES=
set LIB_DX_BASECLASSES_DEBUG=
set LIB_DX_BASECLASSES_RELEASE=

rem endlocal
echo on
