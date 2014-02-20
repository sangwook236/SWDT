#if defined(_WIN32)

#	if defined(_MSC_VER)

#		if defined(DEBUG) || defined(_DEBUG)

//#pragma comment(lib, "surfWinDLL_d.lib")
#pragma comment(lib, "opensift_d.lib")
#pragma comment(lib, "SiftGPU_d.lib")

#pragma comment(lib, "opencv_calib3d243d.lib")
#pragma comment(lib, "opencv_objdetect243d.lib")
#pragma comment(lib, "opencv_imgproc243d.lib")
#pragma comment(lib, "opencv_highgui243d.lib")
#pragma comment(lib, "opencv_core243d.lib")

#pragma comment(lib, "cudpp32d.lib")
#pragma comment(lib, "cudart.lib")

#pragma comment(lib, "libmat.lib")
#pragma comment(lib, "libmx.lib")

#pragma comment(lib, "lapack_d.lib")
#pragma comment(lib, "blas_d.lib")
#pragma comment(lib, "libf2c_d.lib")

#		else

//#pragma comment(lib, "surfWinDLL.lib")
#pragma comment(lib, "opensift.lib")
#pragma comment(lib, "SiftGPU.lib")

#pragma comment(lib, "opencv_calib3d243.lib")
#pragma comment(lib, "opencv_objdetect243.lib")
#pragma comment(lib, "opencv_imgproc243.lib")
#pragma comment(lib, "opencv_highgui243.lib")
#pragma comment(lib, "opencv_core243.lib")

#pragma comment(lib, "cudpp32.lib")
#pragma comment(lib, "cudart.lib")

#pragma comment(lib, "libmat.lib")
#pragma comment(lib, "libmx.lib")

#pragma comment(lib, "lapack.lib")
#pragma comment(lib, "blas.lib")
#pragma comment(lib, "libf2c.lib")

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
