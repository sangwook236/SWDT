#include <iostream>
#include <stdexcept>


int main(void)
{
	int gpio_main();

	int retval = EXIT_SUCCESS;
	try
	{
		retval = gpio_main();
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