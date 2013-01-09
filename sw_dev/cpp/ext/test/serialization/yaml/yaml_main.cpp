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

}  // namespace my_yaml

int yaml_main(int argc, char *argv[])
{
	try
	{
		//my_yaml::basic_parsing();
		my_yaml::basic_emitting();
	}
	catch (const YAML::RepresentationException &e)
	{
		std::cout << "YAML::RepresentationException occurred: " << e.what() << std::endl;
	}
	catch (const YAML::ParserException &e)
	{
		std::cout << "YAML::ParserException occurred: " << e.what() << std::endl;
	}
	catch (const YAML::EmitterException &e)
	{
		std::cout << "YAML::EmitterException occurred: " << e.what() << std::endl;
	}
	catch (const YAML::Exception &e)
	{
		std::cout << "YAML::Exception occurred: " << e.what() << std::endl;
	}

	return 0;
}
