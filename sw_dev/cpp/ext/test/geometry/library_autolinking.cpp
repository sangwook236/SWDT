#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)

#	if defined(_MSC_VER)

#		if defined(DEBUG) || defined(_DEBUG)

#pragma comment(lib, "OpenMeshCored.lib")
#pragma comment(lib, "OpenMeshToolsd.lib")

#pragma comment(lib, "libgmp-10.lib")
//#pragma comment(lib, "libmpfr-4.lib")
//#pragma comment(lib, "libtaucs.lib")

#		else

#pragma comment(lib, "OpenMeshCore.lib")
#pragma comment(lib, "OpenMeshTools.lib")

#pragma comment(lib, "libgmp-10.lib")
//#pragma comment(lib, "libmpfr-4.lib")
//#pragma comment(lib, "libtaucs.lib")

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
