#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)

#	if defined(_MSC_VER)

#		if defined(DEBUG) || defined(_DEBUG)

#pragma comment(lib, "pcl_common_debug.lib")
#pragma comment(lib, "pcl_io_debug.lib")
#pragma comment(lib, "pcl_kdtree_debug.lib")
#pragma comment(lib, "pcl_search_debug.lib")
#pragma comment(lib, "pcl_surface_debug.lib")
#pragma comment(lib, "pcl_features_debug.lib")
#pragma comment(lib, "pcl_visualization_debug.lib")
#pragma comment(lib, "pcl_sample_consensus_debug.lib")

#pragma comment(lib, "vtkCommonCore-7.0.lib")
#pragma comment(lib, "vtkCommonMath-7.0.lib")
#pragma comment(lib, "vtkFiltersCore-7.0.lib")

#pragma comment(lib, "flann.lib")

#		else

#pragma comment(lib, "pcl_common_release.lib")
#pragma comment(lib, "pcl_io_release.lib")
#pragma comment(lib, "pcl_kdtree_release.lib")
#pragma comment(lib, "pcl_search_release.lib")
#pragma comment(lib, "pcl_surface_release.lib")
#pragma comment(lib, "pcl_features_release.lib")
#pragma comment(lib, "pcl_visualization_release.lib")
#pragma comment(lib, "pcl_sample_consensus_release.lib")

#pragma comment(lib, "vtkCommonCore-7.0.lib")
#pragma comment(lib, "vtkCommonMath-7.0.lib")
#pragma comment(lib, "vtkCommonDataModel-7.0.lib")
#pragma comment(lib, "vtkCommonExecutionModel-7.0.lib")
#pragma comment(lib, "vtkRenderingCore-7.0.lib")
#pragma comment(lib, "vtkRenderingLOD-7.0.lib")
#pragma comment(lib, "vtkFiltersCore-7.0.lib")
#pragma comment(lib, "vtkFiltersSources-7.0.lib")

#pragma comment(lib, "flann.lib")

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
