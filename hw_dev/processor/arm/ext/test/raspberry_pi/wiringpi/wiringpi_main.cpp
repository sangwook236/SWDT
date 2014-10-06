#include <wiringPi.h>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_wiringpi {

void gpio();

}  // namespace wiringpi
int wiringpi_main(int argc, char **argv)
{
    // Initialize BCM2835.
    if (!wiringPiSetup())
    {
        std::cerr << "wiringPI library not initialized" << std::endl;
        return 1;
    }

	my_wiringpi::gpio();

    // Close BCM2835.
    bcm2835_close();

	return retval;
}
