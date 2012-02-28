#include <iostream>


#if defined(_UNICODE) || defined(UNICODE)
int wmain(int argc, wchar_t* argv[])
#else
int main(int argc, char* argv[])
#endif
{
	void resampling();
	void greedy_projection();
	void pcl_visualizion(int argc, char **argv);

	try
	{
		// tutorials
		//resampling();
		greedy_projection();

		//pcl_visualizion(argc, argv);
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
