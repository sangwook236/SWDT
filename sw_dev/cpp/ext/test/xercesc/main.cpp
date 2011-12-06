#include "stdafx.h"
#include <iostream>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


#if defined(_UNICODE) || defined(UNICODE)
int wmain(int argc, wchar_t* argv[])
#else
int main(int argc, char* argv[])
#endif
{
	int sax();
	int dom();

	sax();
	dom();

	std::cout << "press any key to terminate" << std::flush;
	std::cin.get();

    return 0;
}
