//#include "stdafx.h"
#if defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <ctime>

#if defined(UNICODE) || defined(_UNICODE)
int wmain(int argc, wchar_t **argv)
#else
int main(int argc, char **argv)
#endif
{
	void k_means();
	void k_medoids();

	try
	{
		std::srand((unsigned int)time(NULL));

		k_means();  // not yet implemented
		k_medoids();  // not yet implemented
	}
	catch (const std::exception &e)
	{
		std::cout << "std::exception occurred !!!: " << e.what() << std::endl;
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
