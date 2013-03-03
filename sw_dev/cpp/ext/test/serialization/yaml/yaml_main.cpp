//#include "stdafx.h"
#include <yaml-cpp/yaml.h>
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_yaml {

void basic_parsing();
void basic_emitting();

void configuration_example();

}  // namespace my_yaml

int yaml_main(int argc, char *argv[])
{
	try
	{
		//my_yaml::basic_parsing();  // for the old API
		//my_yaml::basic_emitting();  // for the old API

		my_yaml::configuration_example();
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
