#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)

#	if defined(_MSC_VER)

#		if defined(DEBUG) || defined(_DEBUG)

#pragma comment(lib, "PartsBasedDetector_d.lib")
#pragma comment(lib, "opencv_calib3d400d.lib")
#pragma comment(lib, "opencv_contrib400d.lib")
#pragma comment(lib, "opencv_core400d.lib")
#pragma comment(lib, "opencv_features2d400d.lib")
#pragma comment(lib, "opencv_flann400d.lib")
#pragma comment(lib, "opencv_gpu400d.lib")
#pragma comment(lib, "opencv_highgui400d.lib")
#pragma comment(lib, "opencv_imgproc400d.lib")
#pragma comment(lib, "opencv_legacy400d.lib")
#pragma comment(lib, "opencv_ml400d.lib")
#pragma comment(lib, "opencv_nonfree400d.lib")
#pragma comment(lib, "opencv_objdetect400d.lib")
#pragma comment(lib, "opencv_photo400d.lib")
#pragma comment(lib, "opencv_stitching400d.lib")
#pragma comment(lib, "opencv_ts400d.lib")
#pragma comment(lib, "opencv_video400d.lib")
#pragma comment(lib, "opencv_videostab400d.lib")
#pragma comment(lib, "cvmatio.lib")

#		else

#pragma comment(lib, "PartsBasedDetector.lib")
#pragma comment(lib, "opencv_calib3d400.lib")
#pragma comment(lib, "opencv_contrib400.lib")
#pragma comment(lib, "opencv_core400.lib")
#pragma comment(lib, "opencv_features2d400.lib")
#pragma comment(lib, "opencv_flann400.lib")
#pragma comment(lib, "opencv_gpu400.lib")
#pragma comment(lib, "opencv_highgui400.lib")
#pragma comment(lib, "opencv_imgproc400.lib")
#pragma comment(lib, "opencv_legacy400.lib")
#pragma comment(lib, "opencv_ml400.lib")
#pragma comment(lib, "opencv_nonfree400.lib")
#pragma comment(lib, "opencv_objdetect400.lib")
#pragma comment(lib, "opencv_photo400.lib")
#pragma comment(lib, "opencv_stitching400.lib")
#pragma comment(lib, "opencv_ts400.lib")
#pragma comment(lib, "opencv_video400.lib")
#pragma comment(lib, "opencv_videostab400.lib")

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
