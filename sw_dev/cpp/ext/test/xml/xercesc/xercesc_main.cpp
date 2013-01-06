//#include "stdafx.h"
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_xercesc {

int sax();
int dom();

}  // namespace my_xercesc

int xercesc_main(int argc, char *argv[])
{
	//my_xercesc::sax();  // not yet implemented
	my_xercesc::dom();

	return 0;
}
