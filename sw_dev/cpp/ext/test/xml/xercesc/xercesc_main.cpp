//#include "stdafx.h"
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_xercesc {

void sax();
void dom();

}  // namespace my_xercesc

int xercesc_main(int argc, char *argv[])
{
	//my_xercesc::sax();  // Not yet implemented.
	my_xercesc::dom();

	return 0;
}
