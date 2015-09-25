#if defined(_WIN32)

#	if defined(_MSC_VER)

#		if defined(DEBUG) || defined(_DEBUG)

#pragma comment(lib, "libGClassesDbg.lib")
#pragma comment(lib, "hdf5_hld.lib")
#pragma comment(lib, "hdf5d.lib")
#pragma comment(lib, "lapack.lib")
#pragma comment(lib, "cblas.lib")
#pragma comment(lib, "blas.lib")
#pragma comment(lib, "libprotobufd.lib")
#pragma comment(lib, "gflags.lib")
#pragma comment(lib, "libglog_d.lib")
//#pragma comment(lib, "libglog_static_d.lib")
#pragma comment(lib, "snappy32.lib")
#pragma comment(lib, "libleveldb.lib")
#pragma comment(lib, "liblmdb.lib")
//#pragma comment(lib, "MultiBoostLib_d.lib")
#pragma comment(lib, "vl.lib")
//#pragma comment(lib, "libshogun-13.0.dll.lib")
//#pragma comment(lib, "librlglue.lib")
//#pragma comment(lib, "librlutils.lib")

#		else

#pragma comment(lib, "libGClasses.lib")
#pragma comment(lib, "hdf5_hl.lib")
#pragma comment(lib, "hdf5.lib")
#pragma comment(lib, "lapack.lib")
#pragma comment(lib, "cblas.lib")
#pragma comment(lib, "blas.lib")
#pragma comment(lib, "libprotobuf.lib")
#pragma comment(lib, "gflags.lib")
#pragma comment(lib, "libglog.lib")
//#pragma comment(lib, "libglog_static.lib")
#pragma comment(lib, "snappy32.lib")
#pragma comment(lib, "libleveldb.lib")
#pragma comment(lib, "liblmdb.lib")
//#pragma comment(lib, "MultiBoostLib.lib")
#pragma comment(lib, "vl.lib")
//#pragma comment(lib, "libshogun-13.0.dll.lib")
//#pragma comment(lib, "librlglue.lib")
//#pragma comment(lib, "librlutils.lib")

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
