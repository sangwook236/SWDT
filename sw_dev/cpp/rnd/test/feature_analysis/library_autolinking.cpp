#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)

#	if defined(_MSC_VER)

#		if defined(DEBUG) || defined(_DEBUG)

//#pragma comment(lib, "surfWinDLL_d.lib")
//#pragma comment(lib, "opensift_d.lib")
//#pragma comment(lib, "SiftGPU_d.lib")

#pragma comment(lib, "opencv_videoio410d.lib")
#pragma comment(lib, "opencv_calib3d410d.lib")
#pragma comment(lib, "opencv_objdetect410d.lib")
#pragma comment(lib, "opencv_imgcodecs410d.lib")
#pragma comment(lib, "opencv_imgproc410d.lib")
#pragma comment(lib, "opencv_highgui410d.lib")
#pragma comment(lib, "opencv_core410d.lib")

//#pragma comment(lib, "cudpp32d.lib")
//#pragma comment(lib, "cudart.lib")

//#pragma comment(lib, "libmat.lib")
//#pragma comment(lib, "libmx.lib")

#pragma comment(lib, "boost_container-vc141-mt-gd-x64-1_67.lib")

#		else

//#pragma comment(lib, "surfWinDLL.lib")
//#pragma comment(lib, "opensift.lib")
//#pragma comment(lib, "SiftGPU.lib")

#pragma comment(lib, "opencv_videoio410.lib")
#pragma comment(lib, "opencv_calib3d410.lib")
#pragma comment(lib, "opencv_objdetect410.lib")
#pragma comment(lib, "opencv_imgcodecs410.lib")
#pragma comment(lib, "opencv_imgproc410.lib")
#pragma comment(lib, "opencv_highgui410.lib")
#pragma comment(lib, "opencv_core410.lib")

//#pragma comment(lib, "cudpp32.lib")
//#pragma comment(lib, "cudart.lib")

//#pragma comment(lib, "libmat.lib")
//#pragma comment(lib, "libmx.lib")

#pragma comment(lib, "boost_chrono-vc141-mt-x64-1_67.lib")

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
