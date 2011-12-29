#include <Python.h>
#include <iostream>


#if defined(_UNICODE) || defined(UNICODE)
int wmain(int argc, wchar_t* argv[])
#else
int main(int argc, char* argv[])
#endif
{
	void embedding_simple_string();
#if defined(_UNICODE) || defined(UNICODE)
	bool embedding_simple_script(int argc, wchar_t* argv[]);
#else
	bool embedding_simple_script(int argc, char* argv[]);
#endif

	try
	{
		//embedding_simple_string();
		const bool retval = embedding_simple_script(argc, argv);
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
