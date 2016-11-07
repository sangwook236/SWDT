#if defined(_WIN32)

#	if defined(_MSC_VER)

#		if defined(DEBUG) || defined(_DEBUG)

//#pragma comment(lib, "libmgl2-glut.lib")
//#pragma comment(lib, "libmgl2.lib")
//#pragma comment(lib, "libmgl-qt.lib")
//#pragma comment(lib, "libmgl-glut.lib")
//#pragma comment(lib, "libmgl.lib")
#pragma comment(lib, "mgl-glutd.lib")
#pragma comment(lib, "mgld.lib")
//#pragma comment(lib, "zlibd.lib")

#pragma comment(lib, "plplot.lib")
#pragma comment(lib, "plplotcxx.lib")
#pragma comment(lib, "csirocsa.lib")
#pragma comment(lib, "qsastime.lib")

#		else

//#pragma comment(lib, "libmgl2-glut.lib")
//#pragma comment(lib, "libmgl2.lib")
//#pragma comment(lib, "libmgl-qt.lib")
//#pragma comment(lib, "libmgl-glut.lib")
//#pragma comment(lib, "libmgl.lib")
#pragma comment(lib, "mgl-glut.lib")
#pragma comment(lib, "mgl.lib")
//#pragma comment(lib, "zlib.lib")

#pragma comment(lib, "plplot.lib")
#pragma comment(lib, "plplotcxx.lib")
#pragma comment(lib, "csirocsa.lib")
#pragma comment(lib, "qsastime.lib")

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
