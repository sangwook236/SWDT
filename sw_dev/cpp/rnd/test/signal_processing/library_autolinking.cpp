#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)

#	if defined(_MSC_VER)

#		if defined(DEBUG) || defined(_DEBUG)

#pragma comment(lib, "Aquilad.lib")
#pragma comment(lib, "Ooura_fftd.lib")
#pragma comment(lib, "spucd.lib")
#pragma comment(lib, "spuc_typesd.lib")
#pragma comment(lib, "spuced.lib")
#pragma comment(lib, "opencv_imgcodecs320d.lib")
#pragma comment(lib, "opencv_imgproc320d.lib")
#pragma comment(lib, "opencv_highgui320d.lib")
#pragma comment(lib, "opencv_core320d.lib")
#pragma comment(lib, "libfftwf-3.3d.lib")
#pragma comment(lib, "libfftw-3.3d.lib")
#pragma comment(lib, "boost_chrono-vc140-mt-gd-1_64.lib")

#		else

#pragma comment(lib, "Aquila.lib")
#pragma comment(lib, "Ooura_fft.lib")
#pragma comment(lib, "spuc.lib")
#pragma comment(lib, "spuc_types.lib")
#pragma comment(lib, "spuce.lib")
#pragma comment(lib, "opencv_imgcodecs320.lib")
#pragma comment(lib, "opencv_imgproc320.lib")
#pragma comment(lib, "opencv_highgui320.lib")
#pragma comment(lib, "opencv_core320.lib")
#pragma comment(lib, "libfftwf-3.3.lib")
#pragma comment(lib, "libfftw-3.3.lib")
#pragma comment(lib, "boost_chrono-vc140-mt-1_64.lib")

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
