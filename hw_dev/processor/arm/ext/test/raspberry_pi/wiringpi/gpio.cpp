#include <wiringPi.h>
#include <iostream>


// LED_PIN Pin - wiringPi pin 0 is BCM_GPIO 17.
#define	LED_PIN	0

namespace {
namespace local {

// [ref] ${WIRINGPI_HOME}/examples/blink.c.
void blink_example()
{
    // Set the pin to be an output.
    pinMode(LED_PIN, OUTPUT) ;

    std::cout << "start blink ..." << std::endl;
    while (true)
    {
    	// Turn on.
	    digitalWrite(LED_PIN, HIGH);
	    delay(500);  // [msec].

		// Turn off.
	    digitalWrite(LED_PIN, LOW);
	    delay(500);  // [msec].
    }
    std::cout << "end blink ..." << std::endl;
}

}  // namespace local
}  // unnamed namespace

namespace my_wiringpi {

void gpio()
{
    local::blink_example();
}

}  // namespace my_wiringpi
