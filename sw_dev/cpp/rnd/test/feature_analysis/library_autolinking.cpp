#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)

#	if defined(_MSC_VER)

#		if defined(DEBUG) || defined(_DEBUG)

//#pragma comment(lib, "surfWinDLL_d.lib")
//#pragma comment(lib, "opensift_d.lib")
//#pragma comment(lib, "SiftGPU_d.lib")

#pragma comment(lib, "opencv_videoio310d.lib")
#pragma comment(lib, "opencv_calib3d310d.lib")
#pragma comment(lib, "opencv_objdetect310d.lib")
#pragma comment(lib, "opencv_imgcodecs310d.lib")
#pragma comment(lib, "opencv_imgproc310d.lib")
#pragma comment(lib, "opencv_highgui310d.lib")
#pragma comment(lib, "opencv_core310d.lib")

//#pragma comment(lib, "cudpp32d.lib")
//#pragma comment(lib, "cudart.lib")

//#pragma comment(lib, "libmat.lib")
//#pragma comment(lib, "libmx.lib")

#pragma comment(lib, "boost_chrono-vc140-mt-gd-1_61.lib")

#pragma comment(lib, "lapackd.lib")
#pragma comment(lib, "blasd.lib")
#pragma comment(lib, "libf2cd.lib")

#		else

//#pragma comment(lib, "surfWinDLL.lib")
//#pragma comment(lib, "opensift.lib")
//#pragma comment(lib, "SiftGPU.lib")

#pragma comment(lib, "opencv_videoio310.lib")
#pragma comment(lib, "opencv_calib3d310.lib")
#pragma comment(lib, "opencv_objdetect310.lib")
#pragma comment(lib, "opencv_imgcodecs310.lib")
#pragma comment(lib, "opencv_imgproc310.lib")
#pragma comment(lib, "opencv_highgui310.lib")
#pragma comment(lib, "opencv_core310.lib")

//#pragma comment(lib, "cudpp32.lib")
//#pragma comment(lib, "cudart.lib")

//#pragma comment(lib, "libmat.lib")
//#pragma comment(lib, "libmx.lib")

#pragma comment(lib, "boost_chrono-vc140-mt-1_61.lib")

#pragma comment(lib, "lapack.lib")
#pragma comment(lib, "blas.lib")
#pragma comment(lib, "libf2c.lib")

#		endif

#	else

//#error [SWDT] not supported compiler

#	endif

#elif defined(__MINGW32__)

#	if defined(__GUNC__)

#		if defined(DEBUG) || defined(_DEBUG)
#		else
#		endif

#	else

//#error [SWDT] not supported compiler

#	endif

#elif defined(__CYGWIN__)

#	if defined(__GUNC__)

#		if defined(DEBUG) || defined(_DEBUG)
#		else
#		endif

#	else

//#error [SWDT] not supported compiler

#	endif

#elif defined(__unix__) || defined(__unix) || defined(unix) || defined(__linux__) || defined(__linux) || defined(linux)

#	if defined(__GUNC__)

#		if defined(DEBUG) || defined(_DEBUG)
#		else
#		endif

#	else

//#error [SWDT] not supported compiler

#	endif

#elif defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__ ) || defined(__DragonFly__)

#	if defined(__GUNC__)

#		if defined(DEBUG) || defined(_DEBUG)
#		else
#		endif

#	else

//#error [SWDT] not supported compiler

#	endif

#elif defined(__APPLE__)

#	if defined(__GUNC__)

#		if defined(DEBUG) || defined(_DEBUG)
#		else
#		endif

#	else

//#error [SWDT] not supported compiler

#	endif

#else

#error [SWDT] not supported operating sytem

#endif
