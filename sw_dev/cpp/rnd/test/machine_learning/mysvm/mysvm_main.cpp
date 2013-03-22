//#include "stdafx.h"
#include "../mysvm_lib/globals.h"
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_mysvm {

void learn_example();
void predict_example();

}  // namespace my_mysvm

int mysvm_main(int argc, char *argv[])
{
	try
	{
		my_mysvm::learn_example();  // not yet implemented
		my_mysvm::predict_example();  // not yet implemented
	}
	catch (const general_exception &e)
	{
		std::cout << "mySVM's general_exception caught: "<< e.error_msg << std::endl;
		return 1;
	}

	return 0;
}
