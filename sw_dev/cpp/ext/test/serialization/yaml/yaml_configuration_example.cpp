//#include "stdafx.h"
#include <yaml-cpp/yaml.h>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <fstream>
#include <iostream>
#include <string>


namespace {
namespace local {

}  // local
}  // unnamed namespace

namespace my_yaml {

void configuration_example()
{
    YAML::Node config = YAML::LoadFile("serialization_data/yaml/config.yaml");

    if (config["lastLogin"])
        std::cout << "last logged in: " << config["lastLogin"].as<std::string>() << std::endl;

    const std::string username = config["username"].as<std::string>();
    const std::string password = config["password"].as<std::string>();

	boost::posix_time::ptime now = boost::posix_time::second_clock::local_time();
    config["lastLogin"] = boost::posix_time::to_simple_string(now);

    std::ofstream stream("serialization_data/yaml/config.yaml");
    stream << config;
}

}  // namespace my_yaml
