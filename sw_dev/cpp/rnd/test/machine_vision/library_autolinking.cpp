#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)

#	if defined(_MSC_VER)

#		if defined(DEBUG) || defined(_DEBUG)

//#pragma comment(lib, "drwnVisiond.lib")
//#pragma comment(lib, "drwnPGMd.lib")
//#pragma comment(lib, "drwnMLd.lib")
//#pragma comment(lib, "drwnIOd.lib")
//#pragma comment(lib, "drwnBased.lib")

#pragma comment(lib, "opencv_aruco410d.lib")
#pragma comment(lib, "opencv_bgsegm410d.lib")
#pragma comment(lib, "opencv_bioinspired410d.lib")
#pragma comment(lib, "opencv_calib3d410d.lib")
#pragma comment(lib, "opencv_ccalib410d.lib")
#pragma comment(lib, "opencv_core410d.lib")
#pragma comment(lib, "opencv_datasets410d.lib")
#pragma comment(lib, "opencv_dnn410d.lib")
#pragma comment(lib, "opencv_dpm410d.lib")
#pragma comment(lib, "opencv_face410d.lib")
#pragma comment(lib, "opencv_features2d410d.lib")
#pragma comment(lib, "opencv_flann410d.lib")
#pragma comment(lib, "opencv_fuzzy410d.lib")
#pragma comment(lib, "opencv_highgui410d.lib")
#pragma comment(lib, "opencv_imgcodecs410d.lib")
#pragma comment(lib, "opencv_imgproc410d.lib")
#pragma comment(lib, "opencv_line_descriptor410d.lib")
#pragma comment(lib, "opencv_ml410d.lib")
#pragma comment(lib, "opencv_objdetect410d.lib")
#pragma comment(lib, "opencv_optflow410d.lib")
#pragma comment(lib, "opencv_photo410d.lib")
#pragma comment(lib, "opencv_plot410d.lib")
#pragma comment(lib, "opencv_reg410d.lib")
#pragma comment(lib, "opencv_rgbd410d.lib")
#pragma comment(lib, "opencv_saliency410d.lib")
#pragma comment(lib, "opencv_sfm410d.lib")
#pragma comment(lib, "opencv_shape410d.lib")
#pragma comment(lib, "opencv_stereo410d.lib")
#pragma comment(lib, "opencv_stitching410d.lib")
#pragma comment(lib, "opencv_structured_light410d.lib")
#pragma comment(lib, "opencv_superres410d.lib")
#pragma comment(lib, "opencv_surface_matching410d.lib")
#pragma comment(lib, "opencv_text410d.lib")
#pragma comment(lib, "opencv_tracking410d.lib")
//#pragma comment(lib, "opencv_ts410d.lib")
#pragma comment(lib, "opencv_video410d.lib")
#pragma comment(lib, "opencv_videoio410d.lib")
#pragma comment(lib, "opencv_videostab410d.lib")
//#pragma comment(lib, "opencv_viz410d.lib")
#pragma comment(lib, "opencv_xfeatures2d410d.lib")
#pragma comment(lib, "opencv_ximgproc410d.lib")
#pragma comment(lib, "opencv_xobjdetect410d.lib")
#pragma comment(lib, "opencv_xphoto410d.lib")

// Libraries of debug build version must be used.
#pragma comment(lib, "mbld.lib")
#pragma comment(lib, "mcald.lib")
#pragma comment(lib, "fhsd.lib")
#pragma comment(lib, "msm_utilsd.lib")
#pragma comment(lib, "msmd.lib")
#pragma comment(lib, "vil3d_iod.lib")
#pragma comment(lib, "vil3d_algod.lib")
#pragma comment(lib, "vil3dd.lib")
//#pragma comment(lib, "vil1_iod.lib")
#pragma comment(lib, "vil1d.lib")
#pragma comment(lib, "vil_prod.lib")
#pragma comment(lib, "vil_iod.lib")
#pragma comment(lib, "vil_algod.lib")
#pragma comment(lib, "vild.lib")
#pragma comment(lib, "vgl_xiod.lib")
#pragma comment(lib, "vgl_iod.lib")
#pragma comment(lib, "vgl_algod.lib")
#pragma comment(lib, "vgld.lib")
#pragma comment(lib, "vul_iod.lib")
#pragma comment(lib, "vuld.lib")
#pragma comment(lib, "vimtd.lib")
#pragma comment(lib, "vnl_xiod.lib")
#pragma comment(lib, "vnl_iod.lib")
#pragma comment(lib, "vnl_algod.lib")
#pragma comment(lib, "vnld.lib")
#pragma comment(lib, "vsld.lib")
#pragma comment(lib, "vcld.lib")
#pragma comment(lib, "vbl_iod.lib")
#pragma comment(lib, "vbld.lib")
#pragma comment(lib, "v3p_netlibd.lib")
#pragma comment(lib, "netlibd.lib")
#pragma comment(lib, "openjpeg2d.lib")
#pragma comment(lib, "geotiff.lib")

//#pragma comment(lib, "IVTOpenCVLibd.lib")
#pragma comment(lib, "IVTWin32GUILibd.lib")
#pragma comment(lib, "IVTLibd.lib")
#pragma comment(lib, "vl_d.lib")
//#pragma comment(lib, "libccv.lib")  // Run-time error.

#pragma comment(lib, "libboost_chrono-vc141-mt-gd-x64-1_67.lib")
#pragma comment(lib, "libboost_system-vc141-mt-gd-x64-1_67.lib")

#pragma comment(lib, "libjpegd.lib")
#pragma comment(lib, "libpng16d.lib")
#pragma comment(lib, "libtiffd.lib")

#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "Comctl32.lib")
#pragma comment(lib, "Vfw32.lib")
#pragma comment(lib, "ws2_32.lib")

#		else

//#pragma comment(lib, "drwnVision.lib")
//#pragma comment(lib, "drwnPGM.lib")
//#pragma comment(lib, "drwnML.lib")
//#pragma comment(lib, "drwnIO.lib")
//#pragma comment(lib, "drwnBase.lib")

#pragma comment(lib, "opencv_aruco410.lib")
#pragma comment(lib, "opencv_bgsegm410.lib")
#pragma comment(lib, "opencv_bioinspired410.lib")
#pragma comment(lib, "opencv_calib3d410.lib")
#pragma comment(lib, "opencv_ccalib410.lib")
#pragma comment(lib, "opencv_core410.lib")
#pragma comment(lib, "opencv_datasets410.lib")
#pragma comment(lib, "opencv_dnn410.lib")
#pragma comment(lib, "opencv_dpm410.lib")
#pragma comment(lib, "opencv_face410.lib")
#pragma comment(lib, "opencv_features2d410.lib")
#pragma comment(lib, "opencv_flann410.lib")
#pragma comment(lib, "opencv_fuzzy410.lib")
#pragma comment(lib, "opencv_highgui410.lib")
#pragma comment(lib, "opencv_imgcodecs410.lib")
#pragma comment(lib, "opencv_imgproc410.lib")
#pragma comment(lib, "opencv_line_descriptor410.lib")
#pragma comment(lib, "opencv_ml410.lib")
#pragma comment(lib, "opencv_objdetect410.lib")
#pragma comment(lib, "opencv_optflow410.lib")
#pragma comment(lib, "opencv_photo410.lib")
#pragma comment(lib, "opencv_plot410.lib")
#pragma comment(lib, "opencv_reg410.lib")
#pragma comment(lib, "opencv_rgbd410.lib")
#pragma comment(lib, "opencv_saliency410.lib")
#pragma comment(lib, "opencv_sfm410.lib")
#pragma comment(lib, "opencv_shape410.lib")
#pragma comment(lib, "opencv_stereo410.lib")
#pragma comment(lib, "opencv_stitching410.lib")
#pragma comment(lib, "opencv_structured_light410.lib")
#pragma comment(lib, "opencv_superres410.lib")
#pragma comment(lib, "opencv_surface_matching410.lib")
#pragma comment(lib, "opencv_text410.lib")
#pragma comment(lib, "opencv_tracking410.lib")
//#pragma comment(lib, "opencv_ts410.lib")
#pragma comment(lib, "opencv_video410.lib")
#pragma comment(lib, "opencv_videoio410.lib")
#pragma comment(lib, "opencv_videostab410.lib")
//#pragma comment(lib, "opencv_viz410.lib")
#pragma comment(lib, "opencv_xfeatures2d410.lib")
#pragma comment(lib, "opencv_ximgproc410.lib")
#pragma comment(lib, "opencv_xobjdetect410.lib")
#pragma comment(lib, "opencv_xphoto410.lib")

// Libraries of release build version must be used.
#pragma comment(lib, "mbl.lib")
#pragma comment(lib, "mcal.lib")
#pragma comment(lib, "fhs.lib")
#pragma comment(lib, "msm_utils.lib")
#pragma comment(lib, "msm.lib")
#pragma comment(lib, "vil3d_io.lib")
#pragma comment(lib, "vil3d_algo.lib")
#pragma comment(lib, "vil3d.lib")
//#pragma comment(lib, "vil1_io.lib")
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

//#pragma comment(lib, "IVTOpenCVLib.lib")
#pragma comment(lib, "IVTWin32GUILib.lib")
#pragma comment(lib, "IVTLib.lib")
#pragma comment(lib, "vl.lib")
//#pragma comment(lib, "libccv.lib")  // Run-time error.

#pragma comment(lib, "libboost_chrono-vc141-mt-x64-1_67.lib")
#pragma comment(lib, "libboost_system-vc141-mt-x64-1_67.lib")

#pragma comment(lib, "libjpeg.lib")
#pragma comment(lib, "libpng16.lib")
#pragma comment(lib, "libtiff.lib")

#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "Comctl32.lib")
#pragma comment(lib, "Vfw32.lib")
#pragma comment(lib, "ws2_32.lib")

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
