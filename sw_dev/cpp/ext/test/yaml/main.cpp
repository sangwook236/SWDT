#include "stdafx.h"
#include <yaml-cpp/yaml.h>
#include <iostream>


#if defined(_UNICODE) || defined(UNICODE)
int wmain(int argc, wchar_t* argv[])
#else
int main(int argc, char* argv[])
#endif
{
	void basic_parsing();
	void basic_emitting();

	try
	{
		//basic_parsing();
		basic_emitting();
	}
	catch (const YAML::ParserException &e)
	{
		std::cout << "YAML::ParserException occurred: " << e.what() << std::endl;
	}
	catch (const YAML::EmitterException &e)
	{
		std::cout << "YAML::EmitterException occurred: " << e.what() << std::endl;
	}
	catch (const YAML::RepresentationException &e)
	{
		std::cout << "YAML::RepresentationException occurred: " << e.what() << std::endl;
	}
	catch (const YAML::Exception &e)
	{
		std::cout << "YAML::Exception occurred: " << e.what() << std::endl;
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

