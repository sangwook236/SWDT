#if defined(_WIN32)

#	if defined(_MSC_VER)

#		if defined(DEBUG) || defined(_DEBUG)

#pragma comment(lib, "gpd.lib")
#pragma comment(lib, "GClasses_DebugVersion.lib")
#pragma comment(lib, "hdf5_hl_D.lib")
#pragma comment(lib, "hdf5_D.lib")
#pragma comment(lib, "lapackd.lib")
#pragma comment(lib, "gsld.lib")
#pragma comment(lib, "cblasd.lib")
#pragma comment(lib, "blasd.lib")
#pragma comment(lib, "opencv_imgcodecs310d.lib")
#pragma comment(lib, "opencv_imgproc310d.lib")
#pragma comment(lib, "opencv_highgui310d.lib")
#pragma comment(lib, "opencv_core310d.lib")
#pragma comment(lib, "libprotobufd.lib")
#pragma comment(lib, "gflagsd.lib")
#pragma comment(lib, "glogd.lib")
#pragma comment(lib, "snappy32d.lib")
//#pragma comment(lib, "libleveldb.lib")
//#pragma comment(lib, "liblmdb.lib")
//#pragma comment(lib, "MultiBoostLib_d.lib")
//#pragma comment(lib, "vl_d.lib")
//#pragma comment(lib, "libshogun-13.0.dll.lib")
//#pragma comment(lib, "librlglue.lib")
//#pragma comment(lib, "librlutils.lib")
#pragma comment(lib, "libboost_chrono-vc140-mt-gd-1_61.lib")
#pragma comment(lib, "libboost_system-vc140-mt-gd-1_61.lib")

#		else

#pragma comment(lib, "gp.lib")
#pragma comment(lib, "GClasses.lib")
#pragma comment(lib, "hdf5_hl.lib")
#pragma comment(lib, "hdf5.lib")
#pragma comment(lib, "lapack.lib")
#pragma comment(lib, "gsl.lib")
#pragma comment(lib, "cblas.lib")
#pragma comment(lib, "blas.lib")
#pragma comment(lib, "opencv_imgcodecs310.lib")
#pragma comment(lib, "opencv_imgproc310.lib")
#pragma comment(lib, "opencv_highgui310.lib")
#pragma comment(lib, "opencv_core310.lib")
#pragma comment(lib, "libprotobuf.lib")
#pragma comment(lib, "gflags.lib")
#pragma comment(lib, "glog.lib")
#pragma comment(lib, "snappy32.lib")
//#pragma comment(lib, "libleveldb.lib")
//#pragma comment(lib, "liblmdb.lib")
//#pragma comment(lib, "MultiBoostLib.lib")
//#pragma comment(lib, "vl.lib")
//#pragma comment(lib, "libshogun-13.0.dll.lib")
//#pragma comment(lib, "librlglue.lib")
//#pragma comment(lib, "librlutils.lib")
#pragma comment(lib, "libboost_chrono-vc140-mt-1_61.lib")
#pragma comment(lib, "libboost_system-vc140-mt-1_61.lib")

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
