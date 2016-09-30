#if defined(_WIN32)

#	if defined(_MSC_VER)

#		if defined(DEBUG) || defined(_DEBUG)

//#pragma comment(lib, "cvm_ia32_debug.lib")

//#pragma comment(lib, "cblasd.lib")
#pragma comment(lib, "libtatlas.lib")
//#pragma comment(lib, "libsatlas.lib")
#pragma comment(lib, "tmglibd.lib")
#pragma comment(lib, "lapackd.lib")
#pragma comment(lib, "blasd.lib")
#pragma comment(lib, "libf2cd.lib")

#		else

//#pragma comment(lib, "cvm_ia32.lib")

//#pragma comment(lib, "cblas.lib")
#pragma comment(lib, "libtatlas.lib")
//#pragma comment(lib, "libsatlas.lib")
#pragma comment(lib, "tmglib.lib")
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
