#include <Python.h>
#include <iostream>


#if defined(_UNICODE) || defined(UNICODE)
int wmain(int argc, wchar_t* argv[])
#else
int main(int argc, char* argv[])
#endif
{
	void simple_embedding();
#if defined(_UNICODE) || defined(UNICODE)
	bool embedding1(int argc, wchar_t* argv[]);
#else
	bool embedding1(int argc, char* argv[]);
#endif

	try
	{
		bool retval;

		simple_embedding();
		//retval = embedding1(argc, argv);
	}
	catch (const std::exception &e)
	{
		std::cout << "std::exception occurred: " << e.what() << std::endl;
	}
	catch (...)
	{
		std::cout << "unknown exception occurred" << std::endl;
	}

	std::cout << "press any key to exit ..." << std::flush;
	std::cin.get();

    return 0;
}
