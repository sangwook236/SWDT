//#include "stdafx.h"
#include <iostream>


//#include "stdafx.h"
#include <iostream>
#include <stdexcept>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace openni {

void basic_function();
void enumeration_process();

void hand_gesture();
void skeleton();

}  // namespace openni

int openni_main(int argc, char *argv[])
{
	openni::basic_function();
	//openni::enumeration_process();

	//openni::hand_gesture();
	//openni::skeleton();

	return 0;
}
