#define _CRTDBG_MAP_ALLOC
#include <cstdlib>
#include <crtdbg.h>
#include <iostream>


namespace {
namespace local {

void memory_leakage_test()
{
	// References
	//	https://learn.microsoft.com/en-us/cpp/c-runtime-library/find-memory-leaks-using-the-crt-library
	//	https://learn.microsoft.com/en-us/visualstudio/profiling/analyze-memory-usage

	// Including crtdbg.h maps the malloc and free functions to their debug versions, _malloc_dbg and _free_dbg, which track memory allocation and deallocation.

	// To cause an automatic call to _CrtDumpMemoryLeaks at each exit point, place a call to _CrtSetDbgFlag at the beginning of your app with the bit fields
	//_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);

	int* x = (int*)malloc(sizeof(int));
	*x = 7;

	x = (int*)calloc(3, sizeof(int));
	x[0] = 7;
	x[1] = 77;
	x[2] = 777;

	//free(x);  // Memory leak
	x = nullptr;

	int* p = new int(5);
	*p = 10;

	// NOTE [caution] >> It's not reported
	//delete p;  // Memory leak
	p = nullptr;

	// Display a memory-leak report
	std::cout << "----- Memory leak reports." << std::endl;
	// You can use _CrtSetReportMode to redirect the report to another location, or back to the Output window
	//	Report type: _CRT_WARN, _CRT_ERROR, _CRT_ASSERT
	//	Report mode: _CRTDBG_MODE_DEBUG, _CRTDBG_MODE_FILE, _CRTDBG_MODE_WNDW, _CRTDBG_REPORT_MODE
#if 1
	_CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE | _CRTDBG_MODE_WNDW);
	_CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
#else
	_CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_DEBUG);
#endif
	// By default, _CrtDumpMemoryLeaks outputs the memory-leak report to the Debug pane of the Output window.
	_CrtDumpMemoryLeaks();
}

}  // namespace local
}  // unnamed namespace

void resource_leakage()
{
	local::memory_leakage_test();
}
