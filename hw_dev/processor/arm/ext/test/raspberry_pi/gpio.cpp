#include <bcm2835.h>

// Blinks on RPi Plug P1 pin 11 (which is GPIO pin 17)
#define PIN RPI_GPIO_P1_11

namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_gpio {

}  // namespace my_gpio

// [ref] ${BCM2835_HOME}/examples/blink/blink.c.
int gpio_main(int argc, char **argv)
{
    // If you call this, it will not actually access the GPIO.
	//bcm2835_set_debug(1);

	// Initialize BCM2835.
    if (!bcm2835_init())
		return 1;

    // Set the pin to be an output.
    bcm2835_gpio_fsel(PIN, BCM2835_GPIO_FSEL_OUTP);

    while (true)
    {
		// Turn it on.
		bcm2835_gpio_write(PIN, HIGH);

		// Wait a bit.
		bcm2835_delay(500);

		// Turn it off.
		bcm2835_gpio_write(PIN, LOW);

		// Wait a bit.
		bcm2835_delay(500);
    }

	// Initialize BCM2835.
	bcm2835_close();

    return 0;
}
