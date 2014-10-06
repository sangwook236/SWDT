#include <iostream>
#include <stdexcept>
#include <cstdlib>


int main(int argc, char **argv)
{
	int bcm2835_main(int argc, char **argv);
	int wiringpi_main(int argc, char **argv);

	int retval = EXIT_SUCCESS;
	try
	{
		std::cout << "bcm2835 library -----------------------------------------------------" << std::endl;
        retval = bcm2835_main();

		std::cout << "\nwiringPi library ----------------------------------------------------" << std::endl;
        retval = wiringpi_main();
	}
	catch (const std::exception &e)
	{
		std::cout << "std::exception caught: " << e.what() << std::endl;
		retval = EXIT_FAILURE;
	}
	catch (...)
	{
		std::cout << "unknown exception caught" << std::endl;
		retval = EXIT_FAILURE;
	}

	std::cout << "press any key to exit ..." << std::endl;
	std::cin.get();

	return retval;
}
