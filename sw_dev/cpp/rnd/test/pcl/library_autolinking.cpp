#if defined(_WIN32)

#	if defined(_MSC_VER)

#		if defined(DEBUG) || defined(_DEBUG)

#pragma comment(lib, "pcl_common_debug.lib")
#pragma comment(lib, "pcl_io_debug.lib")
#pragma comment(lib, "pcl_kdtree_debug.lib")
#pragma comment(lib, "pcl_search_debug.lib")
#pragma comment(lib, "pcl_surface_debug.lib")
#pragma comment(lib, "pcl_features_debug.lib")
#pragma comment(lib, "pcl_visualization_debug.lib")

#pragma comment(lib, "vtkCommon.lib")
#pragma comment(lib, "vtkFiltering.lib")
#pragma comment(lib, "vtkGraphics.lib")
#pragma comment(lib, "vtkRendering.lib")

#pragma comment(lib, "flann.lib")

#		else

#pragma comment(lib, "pcl_common_release.lib")
#pragma comment(lib, "pcl_io_release.lib")
#pragma comment(lib, "pcl_kdtree_release.lib")
#pragma comment(lib, "pcl_search_release.lib")
#pragma comment(lib, "pcl_surface_release.lib")
#pragma comment(lib, "pcl_features_release.lib")
#pragma comment(lib, "pcl_visualization_release.lib")

#pragma comment(lib, "vtkCommon.lib")
#pragma comment(lib, "vtkFiltering.lib")
#pragma comment(lib, "vtkGraphics.lib")
#pragma comment(lib, "vtkRendering.lib")

#pragma comment(lib, "flann.lib")

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
