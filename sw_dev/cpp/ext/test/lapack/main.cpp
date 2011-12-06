#include <iostream>


int main(int argc, char* argv[])
{
	void cblas();
	void clapack();

	cblas();
	clapack();

	std::cout << "press any key to exit ..." << std::flush;
	std::cin.get();

	return 0;
}

