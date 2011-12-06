#include "stdafx.h"
#include <iostream>


int wmain(int argc, wchar_t* argv[])
{
	void basic();
	void encryption_decryption();
	
	//basic();
	encryption_decryption();

	std::cout << "press any key to terminate" << std::flush;
	std::cin.get();

	return 0;
}