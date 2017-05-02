#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
#include <vld/vld.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
#include <iostream>
#include <cstdlib>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
#if defined(_UNICODE) || defined(UNICODE)
int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPTSTR, int nCmdShow)
#else
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPTSTR, int nCmdShow)
#endif
#else
int main(int argc, char *argv[])
#endif
{
	int qt4_main(int argc, char *argv[]);
	int qt5_main(int argc, char *argv[]);

#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
	int argc = 0;
	char **argv = nullptr;
#endif

	int retval = EXIT_SUCCESS;
	try
	{
		std::cout << "Qt4 library ---------------------------------------------------------" << std::endl;
		//retval = qt4_main(argc, argv);

		std::cout << "\nQt5 library ---------------------------------------------------------" << std::endl;
		retval = qt5_main(argc, argv);

		std::cout << "\nwxWidgets library ---------------------------------------------------" << std::endl;
		// REF [project] >> ${SWDT_C++_HOME}/ext/test/gui/wxwidgets
	}
    catch (const std::bad_alloc &e)
	{
		std::cout << "std::bad_alloc caught: " << e.what() << std::endl;
		retval = EXIT_FAILURE;
	}
	catch (const std::exception &e)
	{
		std::cout << "std::exception caught: " << e.what() << std::endl;
		retval = EXIT_FAILURE;
	}
	catch (...)
	{
		std::cout << "Unknown exception caught" << std::endl;
		retval = EXIT_FAILURE;
	}

	std::cout << "Press any key to exit ..." << std::endl;
	std::cin.get();

	return retval;
}
