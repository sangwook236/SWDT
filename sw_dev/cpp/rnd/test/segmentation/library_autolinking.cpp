#if defined(_WIN32)

#	if defined(_MSC_VER)

#		if defined(DEBUG) || defined(_DEBUG)

//#pragma comment(lib, "opengm_min_sum.lib")
//#pragma comment(lib, "opengm_min_sum_small.lib")
#pragma comment(lib, "external-library-mrfd.lib")
#pragma comment(lib, "external-library-maxflowd.lib")
#pragma comment(lib, "external-library-maxflow-ibfsd.lib")
#pragma comment(lib, "external-library-qpbod.lib")
#pragma comment(lib, "external-library-trwsd.lib")

#pragma comment(lib, "opencv_imgcodecs310d.lib")
#pragma comment(lib, "opencv_imgproc310d.lib")
#pragma comment(lib, "opencv_highgui310d.lib")
#pragma comment(lib, "opencv_core310d.lib")

#pragma comment(lib, "libboost_chrono-vc140-mt-gd-1_61.lib")
#pragma comment(lib, "libboost_system-vc140-mt-gd-1_61.lib")

#		else

//#pragma comment(lib, "opengm_min_sum.lib")
//#pragma comment(lib, "opengm_min_sum_small.lib")
#pragma comment(lib, "external-library-mrf.lib")
#pragma comment(lib, "external-library-maxflow.lib")
#pragma comment(lib, "external-library-maxflow-ibfs.lib")
#pragma comment(lib, "external-library-qpbo.lib")
#pragma comment(lib, "external-library-trws.lib")

#pragma comment(lib, "opencv_imgcodecs310.lib")
#pragma comment(lib, "opencv_imgproc310.lib")
#pragma comment(lib, "opencv_highgui310.lib")
#pragma comment(lib, "opencv_core310.lib")

#pragma comment(lib, "libboost_chrono-vc140-mt-1_61.lib")
#pragma comment(lib, "libboost_system-vc140-mt-1_61.lib")

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
