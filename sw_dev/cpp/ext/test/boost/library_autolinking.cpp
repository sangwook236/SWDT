#if defined(_WIN32)

#	if defined(_MSC_VER)

#		if defined(DEBUG) || defined(_DEBUG)

#pragma comment(lib, "OpenCL.lib")
//#pragma comment(lib, "gmpfrxx_d.lib")
//#pragma comment(lib, "libmpfr-4.dll.lib")
//#pragma comment(lib, "libgmp-3.dll.lib")
#pragma comment(lib, "libboost_graph-vc140-mt-gd-1_61.lib")
#pragma comment(lib, "psapi.lib")  // For Boost.Log library.
#pragma comment(lib, "lapack.lib")
#pragma comment(lib, "blas.lib")
#pragma comment(lib, "libf2c.lib")
#pragma comment(lib, "libjpeg.lib")

#		else

#pragma comment(lib, "OpenCL.lib")
//#pragma comment(lib, "gmpfrxx.lib")
//#pragma comment(lib, "libmpfr-4.dll.lib")
//#pragma comment(lib, "libgmp-3.dll.lib")
#pragma comment(lib, "libboost_graph-vc140-mt-1_61.lib")
#pragma comment(lib, "psapi.lib")  // For Boost.Log library.
#pragma comment(lib, "lapack.lib")
#pragma comment(lib, "blas.lib")
#pragma comment(lib, "libf2c.lib")
#pragma comment(lib, "libjpeg.lib")

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
