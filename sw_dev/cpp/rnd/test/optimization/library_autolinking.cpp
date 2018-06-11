#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)

#	if defined(_MSC_VER)

#		if defined(DEBUG) || defined(_DEBUG)

//#pragma comment(lib, "libscip.mingw.x86.msvc.opt.lib")
//#pragma comment(lib, "libobjscip.mingw.x86.msvc.opt.lib")
//#pragma comment(lib, "libnlpi.cppad.mingw.x86.msvc.opt.lib")
//#pragma comment(lib, "liblpispx2.mingw.x86.msvc.opt.lib")
#pragma comment(lib, "libOsiVold.lib")
#pragma comment(lib, "libVold.lib")
//#pragma comment(lib, "libOsiDylpd.lib")
//#pragma comment(lib, "libDylpd.lib")
#pragma comment(lib, "libOsiSymd.lib")
#pragma comment(lib, "libSymphonyd.lib")
#pragma comment(lib, "libOsiCbcd.lib")
#pragma comment(lib, "libCbcd.lib")
#pragma comment(lib, "libOsiClpd.lib")
#pragma comment(lib, "libClpd.lib")
#pragma comment(lib, "libCgld.lib")
#pragma comment(lib, "libOsid.lib")
#pragma comment(lib, "libCoinUtilsd.lib")
#pragma comment(lib, "levmard.lib")
#pragma comment(lib, "ceres-debug.lib")
#pragma comment(lib, "libcholmodd.lib")
#pragma comment(lib, "libcamdd.lib")
#pragma comment(lib, "libamdd.lib")
#pragma comment(lib, "libccolamdd.lib")
#pragma comment(lib, "libcolamdd.lib")
#pragma comment(lib, "suitesparseconfigd.lib")
#pragma comment(lib, "libspqrd.lib")
#pragma comment(lib, "libcxsparsed.lib")
#pragma comment(lib, "metisd.lib")
#pragma comment(lib, "glogd.lib")
#pragma comment(lib, "gflagsd.lib")
#pragma comment(lib, "shlwapi.lib")
#pragma comment(lib, "glpk_4_60.lib")
#pragma comment(lib, "nlopt.lib")
#pragma comment(lib, "libboost_chrono-vc141-mt-gd-x64-1_67.lib")
#pragma comment(lib, "libboost_system-vc141-mt-gd-x64-1_67.lib")
//#pragma comment(lib, "gsld.lib")
//#pragma comment(lib, "cblasd.lib")

#		else

//#pragma comment(lib, "libscip.mingw.x86.msvc.opt.lib")
//#pragma comment(lib, "libobjscip.mingw.x86.msvc.opt.lib")
//#pragma comment(lib, "libnlpi.cppad.mingw.x86.msvc.opt.lib")
//#pragma comment(lib, "liblpispx2.mingw.x86.msvc.opt.lib")
#pragma comment(lib, "libOsiVol.lib")
#pragma comment(lib, "libVol.lib")
//#pragma comment(lib, "libOsiDylp.lib")
//#pragma comment(lib, "libDylp.lib")
#pragma comment(lib, "libOsiSym.lib")
#pragma comment(lib, "libSymphony.lib")
#pragma comment(lib, "libOsiCbc.lib")
#pragma comment(lib, "libCbc.lib")
#pragma comment(lib, "libOsiClp.lib")
#pragma comment(lib, "libClp.lib")
#pragma comment(lib, "libCgl.lib")
#pragma comment(lib, "libOsi.lib")
#pragma comment(lib, "libCoinUtils.lib")
#pragma comment(lib, "levmar.lib")
#pragma comment(lib, "ceres.lib")
#pragma comment(lib, "libcholmod.lib")
#pragma comment(lib, "libcamd.lib")
#pragma comment(lib, "libamd.lib")
#pragma comment(lib, "libccolamd.lib")
#pragma comment(lib, "libcolamd.lib")
#pragma comment(lib, "suitesparseconfig.lib")
#pragma comment(lib, "libspqr.lib")
#pragma comment(lib, "libcxsparse.lib")
#pragma comment(lib, "metis.lib")
#pragma comment(lib, "glog.lib")
#pragma comment(lib, "gflags.lib")
#pragma comment(lib, "shlwapi.lib")
#pragma comment(lib, "glpk_4_60.lib")
#pragma comment(lib, "nlopt.lib")
#pragma comment(lib, "libboost_chrono-vc141-mt-x64-1_67.lib")
#pragma comment(lib, "libboost_system-vc141-mt-x64-1_67.lib")
//#pragma comment(lib, "gsl.lib")
//#pragma comment(lib, "cblas.lib")

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
