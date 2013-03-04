//#include "stdafx.h"
#include <yaml-cpp/yaml.h>
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_yaml {

#if 0
// for the old APIs
void basic_parsing();
void basic_emitting();
#else
void configuration_example();
void example_0_5();
#endif

}  // namespace my_yaml

int yaml_main(int argc, char *argv[])
{
	try
	{
#if 0
		// for the old APIs
		my_yaml::basic_parsing();
		my_yaml::basic_emitting();
#else
		my_yaml::configuration_example();
		my_yaml::example_0_5();
#endif
	}
	catch (const YAML::RepresentationException &e)
	{
		std::cout << "YAML::RepresentationException caught: " << e.what() << std::endl;
		return 1;
	}
	catch (const YAML::ParserException &e)
	{
		std::cout << "YAML::ParserException caught: " << e.what() << std::endl;
		return 1;
	}
	catch (const YAML::EmitterException &e)
	{
		std::cout << "YAML::EmitterException caught: " << e.what() << std::endl;
		return 1;
	}
	catch (const YAML::Exception &e)
	{
		std::cout << "YAML::Exception caught: " << e.what() << std::endl;
		return 1;
	}

	return 0;
}
