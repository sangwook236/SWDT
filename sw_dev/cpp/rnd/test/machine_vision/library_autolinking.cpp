#if defined(_WIN32)

#	if defined(_MSC_VER)

#		if defined(DEBUG) || defined(_DEBUG)

#pragma comment(lib, "drwnVisiond.lib")
#pragma comment(lib, "drwnPGMd.lib")
#pragma comment(lib, "drwnMLd.lib")
#pragma comment(lib, "drwnIOd.lib")
#pragma comment(lib, "drwnBased.lib")

#pragma comment(lib, "opencv_aruco310d.lib")
#pragma comment(lib, "opencv_bgsegm310d.lib")
#pragma comment(lib, "opencv_bioinspired310d.lib")
#pragma comment(lib, "opencv_calib3d310d.lib")
#pragma comment(lib, "opencv_ccalib310d.lib")
#pragma comment(lib, "opencv_core310d.lib")
#pragma comment(lib, "opencv_datasets310d.lib")
#pragma comment(lib, "opencv_dnn310d.lib")
#pragma comment(lib, "opencv_dpm310d.lib")
#pragma comment(lib, "opencv_face310d.lib")
#pragma comment(lib, "opencv_features2d310d.lib")
#pragma comment(lib, "opencv_flann310d.lib")
#pragma comment(lib, "opencv_fuzzy310d.lib")
#pragma comment(lib, "opencv_highgui310d.lib")
#pragma comment(lib, "opencv_imgcodecs310d.lib")
#pragma comment(lib, "opencv_imgproc310d.lib")
#pragma comment(lib, "opencv_line_descriptor310d.lib")
#pragma comment(lib, "opencv_ml310d.lib")
#pragma comment(lib, "opencv_objdetect310d.lib")
#pragma comment(lib, "opencv_optflow310d.lib")
#pragma comment(lib, "opencv_photo310d.lib")
#pragma comment(lib, "opencv_plot310d.lib")
#pragma comment(lib, "opencv_reg310d.lib")
#pragma comment(lib, "opencv_rgbd310d.lib")
#pragma comment(lib, "opencv_saliency310d.lib")
#pragma comment(lib, "opencv_sfm310d.lib")
#pragma comment(lib, "opencv_shape310d.lib")
#pragma comment(lib, "opencv_stereo310d.lib")
#pragma comment(lib, "opencv_stitching310d.lib")
#pragma comment(lib, "opencv_structured_light310d.lib")
#pragma comment(lib, "opencv_superres310d.lib")
#pragma comment(lib, "opencv_surface_matching310d.lib")
#pragma comment(lib, "opencv_text310d.lib")
#pragma comment(lib, "opencv_tracking310d.lib")
#pragma comment(lib, "opencv_ts310d.lib")
#pragma comment(lib, "opencv_video310d.lib")
#pragma comment(lib, "opencv_videoio310d.lib")
#pragma comment(lib, "opencv_videostab310d.lib")
#pragma comment(lib, "opencv_viz310d.lib")
#pragma comment(lib, "opencv_xfeatures2d310d.lib")
#pragma comment(lib, "opencv_ximgproc310d.lib")
#pragma comment(lib, "opencv_xobjdetect310d.lib")
#pragma comment(lib, "opencv_xphoto310d.lib")

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

#pragma comment(lib, "libjpegd.lib")
#pragma comment(lib, "libpng16d.lib")
#pragma comment(lib, "libtiffd.lib")

#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "Comctl32.lib")
#pragma comment(lib, "Vfw32.lib")
#pragma comment(lib, "ws2_32.lib")

#		else

#pragma comment(lib, "drwnVision.lib")
#pragma comment(lib, "drwnPGM.lib")
#pragma comment(lib, "drwnML.lib")
#pragma comment(lib, "drwnIO.lib")
#pragma comment(lib, "drwnBase.lib")

#pragma comment(lib, "opencv_aruco310.lib")
#pragma comment(lib, "opencv_bgsegm310.lib")
#pragma comment(lib, "opencv_bioinspired310.lib")
#pragma comment(lib, "opencv_calib3d310.lib")
#pragma comment(lib, "opencv_ccalib310.lib")
#pragma comment(lib, "opencv_core310.lib")
#pragma comment(lib, "opencv_datasets310.lib")
#pragma comment(lib, "opencv_dnn310.lib")
#pragma comment(lib, "opencv_dpm310.lib")
#pragma comment(lib, "opencv_face310.lib")
#pragma comment(lib, "opencv_features2d310.lib")
#pragma comment(lib, "opencv_flann310.lib")
#pragma comment(lib, "opencv_fuzzy310.lib")
#pragma comment(lib, "opencv_highgui310.lib")
#pragma comment(lib, "opencv_imgcodecs310.lib")
#pragma comment(lib, "opencv_imgproc310.lib")
#pragma comment(lib, "opencv_line_descriptor310.lib")
#pragma comment(lib, "opencv_ml310.lib")
#pragma comment(lib, "opencv_objdetect310.lib")
#pragma comment(lib, "opencv_optflow310.lib")
#pragma comment(lib, "opencv_photo310.lib")
#pragma comment(lib, "opencv_plot310.lib")
#pragma comment(lib, "opencv_reg310.lib")
#pragma comment(lib, "opencv_rgbd310.lib")
#pragma comment(lib, "opencv_saliency310.lib")
#pragma comment(lib, "opencv_sfm310.lib")
#pragma comment(lib, "opencv_shape310.lib")
#pragma comment(lib, "opencv_stereo310.lib")
#pragma comment(lib, "opencv_stitching310.lib")
#pragma comment(lib, "opencv_structured_light310.lib")
#pragma comment(lib, "opencv_superres310.lib")
#pragma comment(lib, "opencv_surface_matching310.lib")
#pragma comment(lib, "opencv_text310.lib")
#pragma comment(lib, "opencv_tracking310.lib")
#pragma comment(lib, "opencv_ts310.lib")
#pragma comment(lib, "opencv_video310.lib")
#pragma comment(lib, "opencv_videoio310.lib")
#pragma comment(lib, "opencv_videostab310.lib")
#pragma comment(lib, "opencv_viz310.lib")
#pragma comment(lib, "opencv_xfeatures2d310.lib")
#pragma comment(lib, "opencv_ximgproc310.lib")
#pragma comment(lib, "opencv_xobjdetect310.lib")
#pragma comment(lib, "opencv_xphoto310.lib")

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
#pragma comment(lib, "libpng16.lib")
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
