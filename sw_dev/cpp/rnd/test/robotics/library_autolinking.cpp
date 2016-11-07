#if defined(_WIN32)

#	if defined(_MSC_VER)

#		if defined(DEBUG) || defined(_DEBUG)

#pragma comment(lib, "libmrpt-base090-dbg.lib")
#pragma comment(lib, "libmrpt-gui090-dbg.lib")
#pragma comment(lib, "libmrpt-hmtslam090-dbg.lib")
#pragma comment(lib, "libmrpt-hwdrivers090-dbg.lib")
#pragma comment(lib, "libmrpt-maps090-dbg.lib")
#pragma comment(lib, "libmrpt-obs090-dbg.lib")
#pragma comment(lib, "libmrpt-opengl090-dbg.lib")
#pragma comment(lib, "libmrpt-reactivenav090-dbg.lib")
#pragma comment(lib, "libmrpt-slam090-dbg.lib")
#pragma comment(lib, "libmrpt-topography090-dbg.lib")
#pragma comment(lib, "libmrpt-vision090-dbg.lib")

#pragma comment(lib, "ompl_d.lib")
#pragma comment(lib, "libboost_serialization-vc100-mt-gd-1_59.lib")

#		else

#pragma comment(lib, "libmrpt-base090.lib")
#pragma comment(lib, "libmrpt-gui090.lib")
#pragma comment(lib, "libmrpt-hmtslam090.lib")
#pragma comment(lib, "libmrpt-hwdrivers090.lib")
#pragma comment(lib, "libmrpt-maps090.lib")
#pragma comment(lib, "libmrpt-obs090.lib")
#pragma comment(lib, "libmrpt-opengl090.lib")
#pragma comment(lib, "libmrpt-reactivenav090.lib")
#pragma comment(lib, "libmrpt-slam090.lib")
#pragma comment(lib, "libmrpt-topography090.lib")
#pragma comment(lib, "libmrpt-vision090.lib")

#pragma comment(lib, "ompl.lib")
#pragma comment(lib, "libboost_serialization-vc100-mt-1_59.lib")

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
