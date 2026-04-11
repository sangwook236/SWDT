#if defined(_DEBUG) && (defined(_WIN32) || defined(_WIN64))
#define _CRTDBG_MAP_ALLOC
#include <crtdbg.h>
#define new new(_NORMAL_BLOCK, __FILE__, __LINE__)
#endif
#include <cstdlib>
#include <iostream>


namespace {
namespace local {

void memory_leakage_test()
{
	int* x = (int*)malloc(sizeof(int));
	*x = 7;

	//free(x);  // Memory leak
	x = nullptr;

	x = (int*)calloc(3, sizeof(int));
	x[0] = 3;
	x[1] = 33;
	x[2] = 333;

	//free(x);  // Memory leak
	x = nullptr;

	for (int i = 0; i < 3; ++i)
	{
		int* p = new int(5);
		*p = 10;

		//delete p;  // Memory leak
		p = nullptr;
	}
}

}  // namespace local
}  // unnamed namespace

void resource_leakage()
{
#if defined(_DEBUG) && (defined(_WIN32) || defined(_WIN64))
	// References
	//	https://learn.microsoft.com/en-us/cpp/c-runtime-library/find-memory-leaks-using-the-crt-library
	//	https://learn.microsoft.com/en-us/visualstudio/profiling/analyze-memory-usage

	// Including crtdbg.h maps the malloc and free functions to their debug versions, _malloc_dbg and _free_dbg, which track memory allocation and deallocation.

	// To cause an automatic call to _CrtDumpMemoryLeaks at each exit point, place a call to _CrtSetDbgFlag at the beginning of your app with the bit fields
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif

	local::memory_leakage_test();

#if defined(_DEBUG) && (defined(_WIN32) || defined(_WIN64))
	// Display a memory-leak report
	std::cout << "----- Memory leak reports." << std::endl;

	// You can use _CrtSetReportMode to redirect the report to another location, or back to the Output window
	//	Report type: _CRT_WARN, _CRT_ERROR, _CRT_ASSERT
	//	Report mode: _CRTDBG_MODE_DEBUG, _CRTDBG_MODE_FILE, _CRTDBG_MODE_WNDW, _CRTDBG_REPORT_MODE
#if 0
	_CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_DEBUG);  // Default
#else
	_CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_FILE);
	_CrtSetReportFile(_CRT_WARN, _CRTDBG_FILE_STDERR);
#endif
	_CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_DEBUG);  // Default
	_CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE | _CRTDBG_MODE_WNDW);
	_CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);

	// By default, _CrtDumpMemoryLeaks outputs the memory-leak report to the Debug pane of the Output window.
	_CrtDumpMemoryLeaks();  // Not necessary.

	std::cout << "-----" << std::endl;
#endif
}
