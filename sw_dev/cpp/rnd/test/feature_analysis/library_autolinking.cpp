#if defined(_WIN32)

#	if defined(_MSC_VER)

#		if defined(DEBUG) || defined(_DEBUG)

#pragma comment(lib, "sift_libd.lib")
#pragma comment(lib, "SiftGPU_d.lib")
#pragma comment(lib, "surfWinDLL_d.lib")

#pragma comment(lib, "opencv_core243d.lib")
#pragma comment(lib, "opencv_imgproc243d.lib")
#pragma comment(lib, "opencv_highgui243d.lib")

#pragma comment(lib, "libmat.lib")
#pragma comment(lib, "libmx.lib")

#pragma comment(lib, "Qt5Cored.lib")
#pragma comment(lib, "Qt5Guid.lib")

#pragma comment(lib, "gsl.lib")
#pragma comment(lib, "cblas.lib")
#pragma comment(lib, "libprotobufd.lib")
#pragma comment(lib, "libpng15_staticd.lib")
#pragma comment(lib, "zlib.lib")

#		else

#pragma comment(lib, "sift_lib.lib")
#pragma comment(lib, "SiftGPU.lib")
#pragma comment(lib, "surfWinDLL.lib")

#pragma comment(lib, "opencv_core243.lib")
#pragma comment(lib, "opencv_imgproc243.lib")
#pragma comment(lib, "opencv_highgui243.lib")

#pragma comment(lib, "libmat.lib")
#pragma comment(lib, "libmx.lib")

#pragma comment(lib, "Qt5Core.lib")
#pragma comment(lib, "Qt5Gui.lib")

#pragma comment(lib, "gsl.lib")
#pragma comment(lib, "cblas.lib")
#pragma comment(lib, "libprotobuf.lib")
#pragma comment(lib, "libpng15_static.lib")
#pragma comment(lib, "zlib.lib")

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

#elif defined(__linux) || defined(__linux__) || defined(linux) || defined(__unix) || defined(__unix__) || defined(unix)

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
