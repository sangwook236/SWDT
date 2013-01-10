//#include "stdafx.h"
#include <iostream>


//#include "stdafx.h"
#include <iostream>
#include <stdexcept>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_openni {

void basic_function();
void enumeration_process();

void hand_gesture();
void skeleton();

}  // namespace my_openni

int openni_main(int argc, char *argv[])
{
	my_openni::basic_function();
	//my_openni::enumeration_process();

	//my_openni::hand_gesture();
	//my_openni::skeleton();

	return 0;
}
