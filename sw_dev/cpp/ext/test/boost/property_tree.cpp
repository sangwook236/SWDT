#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/info_parser.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <iostream>
#include <stdexcept>


namespace {
namespace local {

void xml_parser()
{
	throw std::runtime_error("not yet implemented");
}

void json_parser()
{
	throw std::runtime_error("not yet implemented");
}

void ini_parser()
{
	throw std::runtime_error("not yet implemented");
}

void info_parser()
{
	throw std::runtime_error("not yet implemented");
}

}  // local
}  // unnamed namespace

void property_tree()
{
	local::xml_parser();
	local::json_parser();
	local::ini_parser();
	local::info_parser();
}
