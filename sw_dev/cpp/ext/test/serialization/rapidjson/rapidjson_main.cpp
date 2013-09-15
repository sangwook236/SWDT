//#include "stdafx.h"
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_rapidjson {

void serialize_example();
void condense_and_pretty_example();

}  // namespace my_rapidjson

int rapidjson_main(int argc, char *argv[])
{
	//my_rapidjson::serialize_example();
	my_rapidjson::condense_and_pretty_example();

    return 0;
}
