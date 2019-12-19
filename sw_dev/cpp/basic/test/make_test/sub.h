#if !defined(__ARITHMETIC_SUB_H__)
#define __ARITHMETIC_SUB_H__ 1

#if defined(BUILD_SHARED_LIB)
#	if defined(ARITHMETIC_LIB_EXPORT)
#		define ARITHMETIC_LIB_API __declspec(dllexport)
#	else
#		define ARITHMETIC_LIB_API __declspec(dllimport)
#	endif
#else
#	define ARITHMETIC_LIB_API
#endif

ARITHMETIC_LIB_API double sub(const double lhs, const double rhs);

#endif  // __ARITHMETIC_SUB_H__
