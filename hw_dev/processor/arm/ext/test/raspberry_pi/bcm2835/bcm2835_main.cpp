#include <bcm2835.h>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_bcm2835 {

void gpio();

}  // namespace my_bcm2835

int bcm2835_main(int argc, char **argv)
{
    // If you call this, it will not actually access the GPIO.
    //bcm2835_set_debug(1);

    // Initialize BCM2835.
    if (!bcm2835_init())
    {
        std::cerr << "BCM2835 library not initialized" << std::endl;
        return 1;
    }

	my_bcm2835::gpio();

    // Close BCM2835.
    bcm2835_close();

	return retval;
}
