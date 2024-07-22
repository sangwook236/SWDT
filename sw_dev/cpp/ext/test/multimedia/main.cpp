#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <stdexcept>
#include <cstdlib>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

int main(int argc, char *argv[])
{
	int portaudio_main(int argc, char *argv[]);
	int sfml_main(int argc, char *argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		std::cout << "PortAudio library ---------------------------------------------------" << std::endl;
		//portaudio_main(argc, argv);  // Not yet implemented.

		std::cout << "\nSimple and Fast Multimedia Library (SFML) ---------------------------" << std::endl;
		//sfml_main(argc, argv);  // Not yet implemented.
	}
	catch (const std::bad_alloc &ex)
	{
		std::cerr << "std::bad_alloc caught: " << ex.what() << std::endl;
		retval = EXIT_FAILURE;
	}
	catch (const std::exception &ex)
	{
		std::cerr << "std::exception caught: " << ex.what() << std::endl;
		retval = EXIT_FAILURE;
	}
	catch (...)
	{
		std::cerr << "Unknown exception caught." << std::endl;
		retval = EXIT_FAILURE;
	}

	std::cout << "Press any key to exit ..." << std::endl;
	std::cin.get();

	return retval;
}
