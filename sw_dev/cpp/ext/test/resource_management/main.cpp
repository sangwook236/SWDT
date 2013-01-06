#include <iostream>


int main(int argc, char *argv[])
{
	int vld_main(int argc, char *argv[]);
	int valgrind_main(int argc, char *argv[]);

	try
	{
		vld_main(argc, argv);
		valgrind_main(argc, argv);
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
