#include <avr/sleep.h>
#include <avr/interrupt.h>


namespace {
namespace local {

void system_init()
{
	/*
	 *	analog comparator
	 */
	ACSR &= ~(_BV(ACIE));  // analog comparator interrupt disable
	ACSR |= _BV(ACD);  // analog comparator disable

	/*
	 *	I/O port
	 */
/*
	// uses all pins on PortA for input
	DDRA = 0x00;
	// it makes port input register(PINn) internally pulled-up state that port output register(PORTn) outputs 1(high)
	PORTA = 0xFF;
	// it makes port input register(PINn) high-impedence state that port output register(PORTn) outputs 0(low)
	// so that we can share the pin with other devices
	//PORTA = 0x00;
*/
	// uses all pins on PortD for output
	DDRD = 0xFF;

	// uses 4 & 7 pins on PortB for PWM
	//DDRB = 0xFF;
	DDRB |= _BV(DDB4);  // PWM 0
	DDRB |= _BV(DDB7);  // PWM 2
}

}  // namespace local
}  // unnamed namespace

namespace my_pwm {
}  // namespace my_pwm

int pwm_main(void)
{
	void timer_init();

	cli();
	local::system_init();
	timer_init();
	sei();

	while (1)
		sleep_mode();

	return 0;
}
