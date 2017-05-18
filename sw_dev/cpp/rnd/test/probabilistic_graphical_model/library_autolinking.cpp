#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)

#	if defined(_MSC_VER)

#		if defined(DEBUG) || defined(_DEBUG)

#pragma comment(lib, "densecrfd.lib")
#pragma comment(lib, "densecrf_optimizationd.lib")  // Rename optimizationd.lib to densecrf_optimizationd.lib.
#pragma comment(lib, "libcrfpp.lib")
#if defined(_WIN64) || defined(WIN64)
#pragma comment(lib, "hCRFd64.lib")
//#pragma comment(lib, "hCRF_ompd64.lib")
#else
#pragma comment(lib, "hCRFd.lib")
//#pragma comment(lib, "hCRF_ompd.lib")
#endif

//#pragma comment(lib, "cygMocapy.dll.lib")  // Not correctly working.
#pragma comment(lib, "external-library-ad3d.lib")
#pragma comment(lib, "external-library-gcod.lib")
#pragma comment(lib, "external-library-maxflowd.lib")
#pragma comment(lib, "external-library-maxflow-ibfsd.lib")
#pragma comment(lib, "external-library-mplpd.lib")
#pragma comment(lib, "external-library-mrfd.lib")
#pragma comment(lib, "external-library-qpbod.lib")
#pragma comment(lib, "external-library-srmpd.lib")
#pragma comment(lib, "external-library-trwsd.lib")
#pragma comment(lib, "libdaid.lib")
//#pragma comment(lib, "pnld.lib")
//#pragma comment(lib, "pnl_cxcored.lib")

#pragma comment(lib, "mpirxxd.lib")
#pragma comment(lib, "mpird.lib")
#pragma comment(lib, "libhdf5_D.lib")

#pragma comment(lib, "opencv_features2d320d.lib")
#pragma comment(lib, "opencv_imgproc320d.lib")
#pragma comment(lib, "opencv_imgcodecs320d.lib")
#pragma comment(lib, "opencv_highgui320d.lib")
#pragma comment(lib, "opencv_core320d.lib")

#pragma comment(lib, "libboost_chrono-vc140-mt-gd-1_64.lib")
#pragma comment(lib, "libboost_system-vc140-mt-gd-1_64.lib")

#		else

#pragma comment(lib, "densecrf.lib")
#pragma comment(lib, "densecrf_optimization.lib")  // Rename optimization.lib to densecrf_optimization.lib.
#pragma comment(lib, "libcrfpp.lib")
#if defined(_WIN64) || defined(WIN64)
#pragma comment(lib, "hCRF64.lib")
//#pragma comment(lib, "hCRF_omp64.lib")
#else
#pragma comment(lib, "hCRF.lib")
//#pragma comment(lib, "hCRF_omp.lib")
#endif

//#pragma comment(lib, "cygMocapy.dll.lib")  // Not correctly working.
#pragma comment(lib, "external-library-ad3.lib")
#pragma comment(lib, "external-library-gco.lib")
#pragma comment(lib, "external-library-maxflow.lib")
#pragma comment(lib, "external-library-maxflow-ibfs.lib")
#pragma comment(lib, "external-library-mplp.lib")
#pragma comment(lib, "external-library-mrf.lib")
#pragma comment(lib, "external-library-qpbo.lib")
#pragma comment(lib, "external-library-srmp.lib")
#pragma comment(lib, "external-library-trws.lib")
#pragma comment(lib, "libdai.lib")
//#pragma comment(lib, "pnl.lib")
//#pragma comment(lib, "pnl_cxcore.lib")

#pragma comment(lib, "mpirxx.lib")
#pragma comment(lib, "mpir.lib")
#pragma comment(lib, "libhdf5.lib")

#pragma comment(lib, "opencv_features2d320.lib")
#pragma comment(lib, "opencv_imgproc320.lib")
#pragma comment(lib, "opencv_imgcodecs320.lib")
#pragma comment(lib, "opencv_highgui320.lib")
#pragma comment(lib, "opencv_core320.lib")

#pragma comment(lib, "libboost_chrono-vc140-mt-1_64.lib")
#pragma comment(lib, "libboost_system-vc140-mt-1_64.lib")

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
