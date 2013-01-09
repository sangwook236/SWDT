//#include "stdafx.h"
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_cairo {

void tutorial();
void drawing_illustration();
void layer_diagram();
void text_extents();

void basic_drawing();

void two_link_arm();

}  // namespace my_cairo

int cairo_main(int argc, char *argv[])
{
	//my_cairo::tutorial();
	//my_cairo::drawing_illustration();  // need to check
	//my_cairo::layer_diagram();  // need to check
	//my_cairo::text_extents();  // need to check

	//my_cairo::basic_drawing();

	//
	my_cairo::two_link_arm();

    return 0;
}
