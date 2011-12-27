#if !defined(__DLL_FUNC__H_)
#define __DLL_FUNC__H_ 1


#if defined(EXPORT_DLL_FUNC)
#	define DLL_FUNC_API __declspec(dllexport)
#else
#	define DLL_FUNC_API __declspec(dllimport)
#endif  // EXPORT_DLL_FUNC


#if defined(__cplusplus)
extern "C" {
#endif

struct DLL_FUNC_API struct_in_dll
{
	int count_;
	int *data_;
};

DLL_FUNC_API int func_in_dll(int i, char *str, struct_in_dll *data);

#if defined(__cplusplus)
}
#endif


#endif  // __DLL_FUNC__H_
