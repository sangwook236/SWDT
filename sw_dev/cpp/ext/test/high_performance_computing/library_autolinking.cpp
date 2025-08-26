#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)

#	if defined(_MSC_VER)

#		if defined(DEBUG) || defined(_DEBUG)

#pragma comment(lib, "cudart.lib")
#pragma comment(lib, "OpenCL.lib")

// For TensorRT
#pragma comment(lib, "nvinfer_10.lib")
#pragma comment(lib, "nvonnxparser_10.lib")

// For ArrayFire
#pragma comment(lib, "af.lib")
//#pragma comment(lib, "afopencl.lib")
//#pragma comment(lib, "afcuda.lib")
//#pragma comment(lib, "afcpu.lib")

#		else

#pragma comment(lib, "cudart.lib")
#pragma comment(lib, "OpenCL.lib")

// For TensorRT
#pragma comment(lib, "nvinfer_10.lib")
#pragma comment(lib, "nvonnxparser_10.lib")

// For ArrayFire
#pragma comment(lib, "af.lib")
//#pragma comment(lib, "afopencl.lib")
//#pragma comment(lib, "afcuda.lib")
//#pragma comment(lib, "afcpu.lib")

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
