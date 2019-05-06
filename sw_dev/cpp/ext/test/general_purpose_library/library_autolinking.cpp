#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)

#	if defined(_MSC_VER)

#		if defined(DEBUG) || defined(_DEBUG)

#pragma comment(lib, "dlib19.17.99_debug_64bit_msvc1913.lib")
//#pragma comment(lib, "stlport.lib")

#if defined(_WIN64) || defined(WIN64)
#pragma comment(lib, "cusolver.lib")
#pragma comment(lib, "cudnn.lib")
#pragma comment(lib, "cublas.lib")
#pragma comment(lib, "curand.lib")
#pragma comment(lib, "cudart.lib")
#endif

#pragma comment(lib, "libpng16d.lib")
#pragma comment(lib, "libjpegd.lib")
#pragma comment(lib, "giflib5d.lib")

#		else

#pragma comment(lib, "dlib19.17.99_release_64bit_msvc1913.lib")
//#pragma comment(lib, "stlport.lib")

#if defined(_WIN64) || defined(WIN64)
#pragma comment(lib, "cusolver.lib")
#pragma comment(lib, "cudnn.lib")
#pragma comment(lib, "cublas.lib")
#pragma comment(lib, "curand.lib")
#pragma comment(lib, "cudart.lib")
#endif

#pragma comment(lib, "libpng16.lib")
#pragma comment(lib, "libjpeg.lib")
#pragma comment(lib, "giflib5.lib")

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

#elif defined(__linux) || defined(__linux__) || defined(linux) || defined(__unix) || defined(__unix__) || defined(unix)

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
