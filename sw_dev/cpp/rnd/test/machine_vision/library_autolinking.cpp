#if defined(_WIN32)

#	if defined(_MSC_VER)

#		if defined(DEBUG) || defined(_DEBUG)

#pragma comment(lib, "opencv_calib3d243d.lib")
#pragma comment(lib, "opencv_contrib243d.lib")
#pragma comment(lib, "opencv_core243d.lib")
#pragma comment(lib, "opencv_features2d243d.lib")
#pragma comment(lib, "opencv_flann243d.lib")
#pragma comment(lib, "opencv_gpu243d.lib")
#pragma comment(lib, "opencv_highgui243d.lib")
#pragma comment(lib, "opencv_imgproc243d.lib")
#pragma comment(lib, "opencv_legacy243d.lib")
#pragma comment(lib, "opencv_ml243d.lib")
#pragma comment(lib, "opencv_nonfree243d.lib")
#pragma comment(lib, "opencv_objdetect243d.lib")
#pragma comment(lib, "opencv_photo243d.lib")
#pragma comment(lib, "opencv_stitching243d.lib")
#pragma comment(lib, "opencv_ts243d.lib")
#pragma comment(lib, "opencv_video243d.lib")
#pragma comment(lib, "opencv_videostab243d.lib")

// libraries of debug build version must be used.
#pragma comment(lib, "mbl.lib")
#pragma comment(lib, "mcal.lib")
#pragma comment(lib, "fhs.lib")
#pragma comment(lib, "msm_utils.lib")
#pragma comment(lib, "msm.lib")
#pragma comment(lib, "vil3d_io.lib")
#pragma comment(lib, "vil3d_algo.lib")
#pragma comment(lib, "vil3d.lib")
#pragma comment(lib, "vil1_io.lib")
#pragma comment(lib, "vil1.lib")
#pragma comment(lib, "vil_pro.lib")
#pragma comment(lib, "vil_io.lib")
#pragma comment(lib, "vil_algo.lib")
#pragma comment(lib, "vil.lib")
#pragma comment(lib, "vgl_xio.lib")
#pragma comment(lib, "vgl_io.lib")
#pragma comment(lib, "vgl_algo.lib")
#pragma comment(lib, "vgl.lib")
#pragma comment(lib, "vul_io.lib")
#pragma comment(lib, "vul.lib")
#pragma comment(lib, "vimt.lib")
#pragma comment(lib, "vnl_xio.lib")
#pragma comment(lib, "vnl_io.lib")
#pragma comment(lib, "vnl_algo.lib")
#pragma comment(lib, "vnl.lib")
#pragma comment(lib, "vsl.lib")
#pragma comment(lib, "vcl.lib")
#pragma comment(lib, "vbl_io.lib")
#pragma comment(lib, "vbl.lib")
#pragma comment(lib, "v3p_netlib.lib")
#pragma comment(lib, "netlib.lib")
#pragma comment(lib, "openjpeg2.lib")
#pragma comment(lib, "geotiff.lib")

#pragma comment(lib, "ivtd.lib")
#pragma comment(lib, "ivtopencvd.lib")
#pragma comment(lib, "ivtwin32guid.lib")
#pragma comment(lib, "vl_d.lib")
//#pragma comment(lib, "libccv.lib")  // run-time error

#pragma comment(lib, "libjpeg.lib")
#pragma comment(lib, "libpng15.lib")
#pragma comment(lib, "libtiff.lib")

#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "Comctl32.lib")
#pragma comment(lib, "Vfw32.lib")
#pragma comment(lib, "ws2_32.lib")

#		else

#pragma comment(lib, "opencv_calib3d243.lib")
#pragma comment(lib, "opencv_contrib243.lib")
#pragma comment(lib, "opencv_core243.lib")
#pragma comment(lib, "opencv_features2d243.lib")
#pragma comment(lib, "opencv_flann243.lib")
#pragma comment(lib, "opencv_gpu243.lib")
#pragma comment(lib, "opencv_highgui243.lib")
#pragma comment(lib, "opencv_imgproc243.lib")
#pragma comment(lib, "opencv_legacy243.lib")
#pragma comment(lib, "opencv_ml243.lib")
#pragma comment(lib, "opencv_nonfree243.lib")
#pragma comment(lib, "opencv_objdetect243.lib")
#pragma comment(lib, "opencv_photo243.lib")
#pragma comment(lib, "opencv_stitching243.lib")
#pragma comment(lib, "opencv_ts243.lib")
#pragma comment(lib, "opencv_video243.lib")
#pragma comment(lib, "opencv_videostab243.lib")

// libraries of release build version must be used.
#pragma comment(lib, "mbl.lib")
#pragma comment(lib, "mcal.lib")
#pragma comment(lib, "fhs.lib")
#pragma comment(lib, "msm_utils.lib")
#pragma comment(lib, "msm.lib")
#pragma comment(lib, "vil3d_io.lib")
#pragma comment(lib, "vil3d_algo.lib")
#pragma comment(lib, "vil3d.lib")
#pragma comment(lib, "vil1_io.lib")
#pragma comment(lib, "vil1.lib")
#pragma comment(lib, "vil_pro.lib")
#pragma comment(lib, "vil_io.lib")
#pragma comment(lib, "vil_algo.lib")
#pragma comment(lib, "vil.lib")
#pragma comment(lib, "vgl_xio.lib")
#pragma comment(lib, "vgl_io.lib")
#pragma comment(lib, "vgl_algo.lib")
#pragma comment(lib, "vgl.lib")
#pragma comment(lib, "vul_io.lib")
#pragma comment(lib, "vul.lib")
#pragma comment(lib, "vimt.lib")
#pragma comment(lib, "vnl_xio.lib")
#pragma comment(lib, "vnl_io.lib")
#pragma comment(lib, "vnl_algo.lib")
#pragma comment(lib, "vnl.lib")
#pragma comment(lib, "vsl.lib")
#pragma comment(lib, "vcl.lib")
#pragma comment(lib, "vbl_io.lib")
#pragma comment(lib, "vbl.lib")
#pragma comment(lib, "v3p_netlib.lib")
#pragma comment(lib, "netlib.lib")
#pragma comment(lib, "openjpeg2.lib")
#pragma comment(lib, "geotiff.lib")

#pragma comment(lib, "ivt.lib")
#pragma comment(lib, "ivtopencv.lib")
#pragma comment(lib, "ivtwin32gui.lib")
#pragma comment(lib, "vl.lib")
//#pragma comment(lib, "libccv.lib")  // run-time error

#pragma comment(lib, "libjpeg.lib")
#pragma comment(lib, "libpng15.lib")
#pragma comment(lib, "libtiff.lib")

#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "Comctl32.lib")
#pragma comment(lib, "Vfw32.lib")
#pragma comment(lib, "ws2_32.lib")

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
