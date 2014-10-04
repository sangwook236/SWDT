#include <bcm2835.h>
#include <iostream>
#include <stdexcept>
#include <cstdlib>


int main(int argc, char **argv)
{
	int gpio_main(int argc, char **argv);

	int retval = EXIT_SUCCESS;
	try
	{
        // If you call this, it will not actually access the GPIO.
        //bcm2835_set_debug(1);

        // Initialize BCM2835.
        if (!bcm2835_init())
        {
            std::cerr << "BCM2835 not initialized" << std::endl;
            return 1;
        }

		retval = gpio_main(argc, argv);

        // Close BCM2835.
        bcm2835_close();
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
