#if !defined(__PartsBasedDetector__EXPORT__H_)
#define __PartsBasedDetector__EXPORT__H_ 1


#if defined(WIN32)
#	if defined(_MSC_VER)
#		if defined(PartsBasedDetector_lib_EXPORTS)
#		    define PartsBasedDetector_API __declspec(dllexport)
#			define PartsBasedDetector_TEMPLATE_EXTERN
#		else
#		    define PartsBasedDetector_API __declspec(dllimport)
#			define PartsBasedDetector_TEMPLATE_EXTERN extern
#		endif  // PartsBasedDetector_lib_EXPORTS
#	else
#		define PartsBasedDetector_API
#		define PartsBasedDetector_TEMPLATE_EXTERN
#	endif  // _MSC_VER
#elif defined(__MINGW32__)
#	if defined(_USRDLL)
#		if defined(PartsBasedDetector_lib_EXPORTS)
#			define PartsBasedDetector_API __declspec(dllexport)
#		else
#			define PartsBasedDetector_API __declspec(dllimport)
#		endif  // PartsBasedDetector_lib_EXPORTS
#	else
#		define PartsBasedDetector_API
#	endif  // _USRDLL
#	define PartsBasedDetector_TEMPLATE_EXTERN
#else
#   define PartsBasedDetector_API
#	define PartsBasedDetector_TEMPLATE_EXTERN
#endif


#endif  // __PartsBasedDetector__EXPORT__H_
