#if defined(_WIN32)

#	if defined(_MSC_VER)

#		if defined(DEBUG) || defined(_DEBUG)

#pragma comment(lib, "libCbc_d.lib")
#pragma comment(lib, "libClp_d.lib")
#pragma comment(lib, "libCgl_d.lib")
#pragma comment(lib, "libOsiClp_d.lib")
#pragma comment(lib, "libOsi_d.lib")
#pragma comment(lib, "libCoinUtils_d.lib")
#pragma comment(lib, "levmar_d.lib")
#pragma comment(lib, "ceres-debug.lib")
#pragma comment(lib, "glog_d.lib")
#pragma comment(lib, "gflags_d.lib")
#pragma comment(lib, "shlwapi.lib")
#pragma comment(lib, "ga_d.lib")
#pragma comment(lib, "glpk_4_57.lib")
#pragma comment(lib, "nlopt.lib")

#		else

#pragma comment(lib, "libCbc.lib")
#pragma comment(lib, "libClp.lib")
#pragma comment(lib, "libCgl.lib")
#pragma comment(lib, "libOsiClp.lib")
#pragma comment(lib, "libOsi.lib")
#pragma comment(lib, "libCoinUtils.lib")
#pragma comment(lib, "levmar.lib")
#pragma comment(lib, "ceres.lib")
#pragma comment(lib, "glog.lib")
#pragma comment(lib, "gflags.lib")
#pragma comment(lib, "shlwapi.lib")
#pragma comment(lib, "ga.lib")
#pragma comment(lib, "glpk_4_57.lib")
#pragma comment(lib, "nlopt.lib")

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
