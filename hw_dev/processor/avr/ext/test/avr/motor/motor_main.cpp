#include "../usart/usart.h"
#include <avr/sleep.h>
#include <avr/interrupt.h>


namespace {
namespace local {

void system_init()
{
	/*
	 *	Analog comparator.
	 */
	ACSR &= ~(_BV(ACIE));  // Analog comparator interrupt disable.
	ACSR |= _BV(ACD);  // Analog comparator disable.

	/*
	 *	I/O port.
	 */
/*
	// Uses all pins on PortA for input.
	DDRA = 0x00;
	// It makes port input register(PINn) internally pulled-up state that port output register(PORTn) outputs 1(high).
	PORTA = 0xFF;
	// It makes port input register(PINn) high-impedence state that port output register(PORTn) outputs 0(low)
	// so that we can share the pin with other devices.
	//PORTA = 0x00;
*/
	// Uses all pins on PortD for output.
	DDRD = 0xFF;

	// Uses 4 & 7 pins on PortB for PWM.
	//DDRB = 0xFF;
	DDRB |= _BV(DDB4);  // PWM 0.
	DDRB |= _BV(DDB7);  // PWM 2.
}

}  // namespace local
}  // unnamed namespace

namespace my_motor {
}  // namespace my_motor

int motor_main(int argc, char *argv[])
{
	void adc_init();
	void timer_init();
	void ext_int_init();

	cli();
	local::system_init();
	adc_init();
	timer_init();
	ext_int_init();
	sei();

	while (1)
		sleep_mode();

	return 0;
}
