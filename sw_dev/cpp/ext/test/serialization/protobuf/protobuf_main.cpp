//#include "stdafx.h"
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_protobuf {

void proto2_example();

}  // namespace my_protobuf

// Usage:
//  protoc --cpp_out=. protobuf_person.proto

int protobuf_main(int argc, char *argv[])
{
	my_protobuf::proto2_example();

    return 0;
}
