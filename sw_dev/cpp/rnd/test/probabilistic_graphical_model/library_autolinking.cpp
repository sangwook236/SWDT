#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)

#	if defined(_MSC_VER)

#		if defined(DEBUG) || defined(_DEBUG)

#pragma comment(lib, "pnld.lib")
#pragma comment(lib, "pnl_cxcored.lib")
//#pragma comment(lib, "cygMocapy.dll.lib")  // not correctly working
#pragma comment(lib, "libdai_d.lib")
#pragma comment(lib, "external-library-ad3d.lib")
#pragma comment(lib, "external-library-gcod.lib")
#pragma comment(lib, "external-library-maxflowd.lib")
#pragma comment(lib, "external-library-maxflow-ibfsd.lib")
#pragma comment(lib, "external-library-mplpd.lib")
#pragma comment(lib, "external-library-mrfd.lib")
#pragma comment(lib, "external-library-qpbod.lib")
#pragma comment(lib, "external-library-srmpd.lib")
#pragma comment(lib, "external-library-trwsd.lib")
#pragma comment(lib, "mpir.lib")
#pragma comment(lib, "hdf5d.lib")

#pragma comment(lib, "crfpp.lib")
#pragma comment(lib, "hcrf_ompd.lib")

#pragma comment(lib, "opencv_features2d310d.lib")
#pragma comment(lib, "opencv_imgproc310d.lib")
#pragma comment(lib, "opencv_highgui310d.lib")
#pragma comment(lib, "opencv_core310d.lib")

#		else

#pragma comment(lib, "pnl.lib")
#pragma comment(lib, "pnl_cxcore.lib")
//#pragma comment(lib, "cygMocapy.dll.lib")  // not correctly working
#pragma comment(lib, "libdai.lib")
#pragma comment(lib, "external-library-ad3.lib")
#pragma comment(lib, "external-library-gco.lib")
#pragma comment(lib, "external-library-maxflow.lib")
#pragma comment(lib, "external-library-maxflow-ibfs.lib")
#pragma comment(lib, "external-library-mplp.lib")
#pragma comment(lib, "external-library-mrf.lib")
#pragma comment(lib, "external-library-qpbo.lib")
#pragma comment(lib, "external-library-srmp.lib")
#pragma comment(lib, "external-library-trws.lib")
#pragma comment(lib, "mpir.lib")
#pragma comment(lib, "hdf5.lib")

#pragma comment(lib, "crfpp.lib")
#pragma comment(lib, "hcrf_omp.lib")

#pragma comment(lib, "opencv_features2d310.lib")
#pragma comment(lib, "opencv_imgproc310.lib")
#pragma comment(lib, "opencv_highgui310.lib")
#pragma comment(lib, "opencv_core310.lib")

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
