#include <AT91SAM7S-EK.h>
#include <AT91SAM7S256.h>
#include <cstdio>


namespace my_at91sam7s {

void delay(unsigned long ms);

}  // namespace my_at91sam7s

namespace {
namespace local {

void system_init()
{
}

void simple_blink_example()
{
    while (true)
    {
        // Switch on the led.
        AT91D_BASE_PIO_LED->PIO_CODR = AT91B_LED1;
        my_at91sam7s::delay(500);

        // Switch off the led.
        AT91D_BASE_PIO_LED->PIO_SODR = AT91B_LED1;
        my_at91sam7s::delay(500);
    }
}

}  // namespace local
}  // unnamed namespace

namespace my_gpio {
}  // namespace my_gpio

int gpio_main(int argc, char *argv[])
{
    // Disable interrupt.
    local::system_init();
    // Enable interrupt.

	local::simple_blink_example();

	return 0;
}
