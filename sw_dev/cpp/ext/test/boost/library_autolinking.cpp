#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)

#	if defined(_MSC_VER)

#		if defined(DEBUG) || defined(_DEBUG)

//#pragma comment(lib, "gmpfrxx_d.lib")
//#pragma comment(lib, "libmpfr-4.dll.lib")
//#pragma comment(lib, "libgmp-3.dll.lib")
#pragma comment(lib, "libboost_graph-vc141-mt-gd-x64-1_67.lib")
//#pragma comment(lib, "libboost_log_setup-vc141-mt-gd-x64-1_67.lib")
//#pragma comment(lib, "libboost_log-vc141-mt-gd-x64-1_67.lib")
#pragma comment(lib, "lapack.lib")
#pragma comment(lib, "blas.lib")
#pragma comment(lib, "libf2c.lib")
#pragma comment(lib, "libjpeg.lib")

#pragma comment(lib, "OpenCL.lib")
#pragma comment(lib, "psapi.lib")  // For Boost.Log library.

#		else

//#pragma comment(lib, "gmpfrxx.lib")
//#pragma comment(lib, "libmpfr-4.dll.lib")
//#pragma comment(lib, "libgmp-3.dll.lib")
#pragma comment(lib, "libboost_graph-vc141-mt-x64-1_67.lib")
//#pragma comment(lib, "libboost_log_setup-vc141-mt-x64-1_67.lib")
//#pragma comment(lib, "libboost_log-vc141-mt-x64-1_67.lib")
#pragma comment(lib, "lapack.lib")
#pragma comment(lib, "blas.lib")
#pragma comment(lib, "libf2c.lib")
#pragma comment(lib, "libjpeg.lib")

#pragma comment(lib, "OpenCL.lib")
#pragma comment(lib, "psapi.lib")  // For Boost.Log library.

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
