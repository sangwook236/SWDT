//#include "stdafx.h"
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace cairo {

void tutorial();
void drawing_illustration();
void layer_diagram();
void text_extents();

void basic_drawing();

void two_link_arm();

}  // namespace cairo

int cairo_main(int argc, char *argv[])
{
	//cairo::tutorial();
	//cairo::drawing_illustration();  // need to check
	//cairo::layer_diagram();  // need to check
	//cairo::text_extents();  // need to check

	//cairo::basic_drawing();

	//
	cairo::two_link_arm();

    return 0;
}
