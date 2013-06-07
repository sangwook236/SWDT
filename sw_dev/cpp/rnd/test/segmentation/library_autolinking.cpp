#if defined(_WIN32)

#	if defined(_MSC_VER)

#		if defined(DEBUG) || defined(_DEBUG)

#pragma comment(lib, "opencv_calib3d243d.lib")
#pragma comment(lib, "opencv_contrib243d.lib")
#pragma comment(lib, "opencv_core243d.lib")
#pragma comment(lib, "opencv_features2d243d.lib")
#pragma comment(lib, "opencv_flann243d.lib")
#pragma comment(lib, "opencv_gpu243d.lib")
#pragma comment(lib, "opencv_highgui243d.lib")
#pragma comment(lib, "opencv_imgproc243d.lib")
#pragma comment(lib, "opencv_legacy243d.lib")
#pragma comment(lib, "opencv_ml243d.lib")
#pragma comment(lib, "opencv_nonfree243d.lib")
#pragma comment(lib, "opencv_objdetect243d.lib")
#pragma comment(lib, "opencv_photo243d.lib")
#pragma comment(lib, "opencv_stitching243d.lib")
#pragma comment(lib, "opencv_ts243d.lib")
#pragma comment(lib, "opencv_video243d.lib")
#pragma comment(lib, "opencv_videostab243d.lib")

#		else

#pragma comment(lib, "opencv_calib3d243.lib")
#pragma comment(lib, "opencv_contrib243.lib")
#pragma comment(lib, "opencv_core243.lib")
#pragma comment(lib, "opencv_features2d243.lib")
#pragma comment(lib, "opencv_flann243.lib")
#pragma comment(lib, "opencv_gpu243.lib")
#pragma comment(lib, "opencv_highgui243.lib")
#pragma comment(lib, "opencv_imgproc243.lib")
#pragma comment(lib, "opencv_legacy243.lib")
#pragma comment(lib, "opencv_ml243.lib")
#pragma comment(lib, "opencv_nonfree243.lib")
#pragma comment(lib, "opencv_objdetect243.lib")
#pragma comment(lib, "opencv_photo243.lib")
#pragma comment(lib, "opencv_stitching243.lib")
#pragma comment(lib, "opencv_ts243.lib")
#pragma comment(lib, "opencv_video243.lib")
#pragma comment(lib, "opencv_videostab243.lib")

#		endif

#	else

#error [SWL] not supported compiler

#	endif

#elif defined(__MINGW32__)

#	if defined(__GUNC__)

#		if defined(DEBUG) || defined(_DEBUG)
#		else
#		endif

#	else

#error [SWL] not supported compiler

#	endif

#elif defined(__CYGWIN__)

#	if defined(__GUNC__)

#		if defined(DEBUG) || defined(_DEBUG)
#		else
#		endif

#	else

#error [SWL] not supported compiler

#	endif

#elif defined(__unix__) || defined(__unix) || defined(unix) || defined(__linux__) || defined(__linux) || defined(linux)

#	if defined(__GUNC__)

#		if defined(DEBUG) || defined(_DEBUG)
#		else
#		endif

#	else

#error [SWL] not supported compiler

#	endif

#elif defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__ ) || defined(__DragonFly__)

#	if defined(__GUNC__)

#		if defined(DEBUG) || defined(_DEBUG)
#		else
#		endif

#	else

#error [SWL] not supported compiler

#	endif

#elif defined(__APPLE__)

#	if defined(__GUNC__)

#		if defined(DEBUG) || defined(_DEBUG)
#		else
#		endif

#	else

#error [SWL] not supported compiler

#	endif

#else

#error [SWL] not supported operating sytem

#endif
