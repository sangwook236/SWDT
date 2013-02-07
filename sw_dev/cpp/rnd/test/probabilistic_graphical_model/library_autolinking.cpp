#if defined(_WIN32)

#	if defined(_MSC_VER)

#		if defined(DEBUG) || defined(_DEBUG)

#pragma comment(lib, "pnld.lib")
#pragma comment(lib, "pnl_cxcored.lib")
#pragma comment(lib, "libdai_d.lib")
#pragma comment(lib, "opengm_min_sum.lib")
#pragma comment(lib, "opengm_min_sum_small.lib")
#pragma comment(lib, "external-library-mrf.lib")
#pragma comment(lib, "external-library-maxflow.lib")
#pragma comment(lib, "external-library-maxflow-ibfs.lib")
#pragma comment(lib, "external-library-qpbo.lib")
#pragma comment(lib, "external-library-trws.lib")

#pragma comment(lib, "crfpp.lib")
#pragma comment(lib, "hcrf_ompd.lib")

#		else

#pragma comment(lib, "pnl.lib")
#pragma comment(lib, "pnl_cxcore.lib")
#pragma comment(lib, "libdai.lib")
#pragma comment(lib, "external-library-mrf.lib")
#pragma comment(lib, "external-library-maxflow.lib")
#pragma comment(lib, "external-library-maxflow-ibfs.lib")
#pragma comment(lib, "external-library-qpbo.lib")
#pragma comment(lib, "external-library-trws.lib")

#pragma comment(lib, "crfpp.lib")
#pragma comment(lib, "hcrf_omp.lib")

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
