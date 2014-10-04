#include <bcm2835.h>
#include <iostream>


// Blinks on RPi Plug P1 pin 11 (which is GPIO pin 17).
#define PIN RPI_GPIO_P1_11

namespace {
namespace local {

// [ref] ${BCM2835_HOME}/examples/blink/blink.c.
void blink_example()
{
    // Set the pin to be an output.
    bcm2835_gpio_fsel(PIN, BCM2835_GPIO_FSEL_OUTP);

    std::cout << "start blink ..." << std::endl;
    while (true)
    {
		// Turn it on.
		bcm2835_gpio_write(PIN, HIGH);

		// Wait a bit.
		bcm2835_delay(500);  // [msec].

		// Turn it off.
		bcm2835_gpio_write(PIN, LOW);

		// Wait a bit.
		bcm2835_delay(500);  // [msec].
    }
    std::cout << "end blink ..." << std::endl;
}

}  // namespace local
}  // unnamed namespace

namespace my_gpio {

}  // namespace my_gpio

int gpio_main(int argc, char **argv)
{
    local::blink_example();

    return 0;
}
