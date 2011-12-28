#include <cuda_runtime.h>
#include <iostream>


#if defined(__cplusplus)
extern "C" {
#endif

//__global__ void HelloWorld();
__global__ void HelloWorld()
{
	std::cout << "Hello World!" << std::endl;
}

#if defined(UNICODE) || defined(_UNICODE)
int wmain(int argc, wchar_t **argv)
#else
int main(int argc, char **argv)
#endif
{

	try
	{
		HelloWorld();
	}
	catch (const std::exception &e)
	{
		std::wcout << L"exception occurred !!!: " << e.what() << std::endl;
	}
	catch (...)
	{
		std::wcout << L"unknown exception occurred !!!" << std::endl;
	}

	std::wcout << L"press any key to exit ..." << std::endl;
	std::wcout.flush();
	std::wcin.get();

    return 0;
}

#if defined(__cplusplus)
}  // extern "C"
#endif
