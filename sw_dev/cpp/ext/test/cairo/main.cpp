//#include "stdafx.h"
#if defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>


int main(int argc, char *argv[])
{
	void tutorial();
	void drawing_illustration();
	void layer_diagram();
	void text_extents();

	void basic_drawing();

	void two_link_arm();

	try
	{
		//tutorial();
		//drawing_illustration();  // need to check
		//layer_diagram();  // need to check
		//text_extents();  // need to check

		//basic_drawing();

		//
		two_link_arm();
	}
	catch (const std::exception &e)
	{
		std::cout << "std::exception occurred: " << e.what() << std::endl;
	}
	catch (...)
	{
		std::cout << "unknown exception occurred" << std::endl;
	}

	std::cout << "press any key to exit ..." << std::flush;
	std::cin.get();

    return 0;
}
