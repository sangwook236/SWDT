#if defined(_WIN32)

#	if defined(_MSC_VER)

#		if defined(DEBUG) || defined(_DEBUG)

//#pragma comment(lib, "viennacld.lib")
//#pragma comment(lib, "cvm_ia32_debug.lib")
#pragma comment(lib, "libsuperlu.lib")
//#pragma comment(lib, "libumfpackd.lib")
//#pragma comment(lib, "libspqrd.lib")
//#pragma comment(lib, "libldld.lib")
//#pragma comment(lib, "libklud.lib")
//#pragma comment(lib, "libcxsparsed.lib")
//#pragma comment(lib, "libcholmodd.lib")
//#pragma comment(lib, "libbtfd.lib")
//#pragma comment(lib, "libcolamdd.lib")
//#pragma comment(lib, "libccolamdd.lib")
//#pragma comment(lib, "libcamdd.lib")
//#pragma comment(lib, "libamdd.lib")
//#pragma comment(lib, "suitesparseconfigd.lib")
#pragma comment(lib, "libtatlas.lib")
//#pragma comment(lib, "libsatlas.lib")
//#pragma comment(lib, "cblasd.lib")
#pragma comment(lib, "libopenblas.lib")
#pragma comment(lib, "tmglibd.lib")
#pragma comment(lib, "lapackd.lib")
#pragma comment(lib, "blasd.lib")
#pragma comment(lib, "libf2cd.lib")
#pragma comment(lib, "OpenCL.lib")

#		else

//#pragma comment(lib, "viennacl.lib")
//#pragma comment(lib, "cvm_ia32.lib")
#pragma comment(lib, "libsuperlu.lib")
//#pragma comment(lib, "libumfpack.lib")
//#pragma comment(lib, "libspqr.lib")
//#pragma comment(lib, "libldl.lib")
//#pragma comment(lib, "libklu.lib")
//#pragma comment(lib, "libcxsparse.lib")
//#pragma comment(lib, "libcholmod.lib")
//#pragma comment(lib, "libbtf.lib")
//#pragma comment(lib, "libcolamd.lib")
//#pragma comment(lib, "libccolamd.lib")
//#pragma comment(lib, "libcamd.lib")
//#pragma comment(lib, "libamd.lib")
//#pragma comment(lib, "suitesparseconfig.lib")
#pragma comment(lib, "libtatlas.lib")
//#pragma comment(lib, "libsatlas.lib")
//#pragma comment(lib, "cblas.lib")
#pragma comment(lib, "libopenblas.lib")
#pragma comment(lib, "tmglib.lib")
#pragma comment(lib, "lapack.lib")
#pragma comment(lib, "blas.lib")
#pragma comment(lib, "libf2c.lib")
#pragma comment(lib, "OpenCL.lib")

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
