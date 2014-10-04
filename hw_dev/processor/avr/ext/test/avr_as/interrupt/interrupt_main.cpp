#include <avr/sleep.h>
#include <avr/interrupt.h>
#include <util/delay.h>


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
	// uses all pins on PortA for output
	DDRA = 0xFF;

	// uses all pins on PortD for input
	DDRD = 0x00;
	PORTD = 0xFF;  // unnecessary
}

}  // namespace local
}  // unnamed namespace

namespace my_intrrupt {
}  // namespace my_intrrupt

int intrrupt_main(void)
{
	void timer_init();
	void ext_int_init();

	cli();
	///EIFR = 0x00;
	local::system_init();
	//timer_init();
	ext_int_init();
	//EIFR = 0x00;
	sei();

	_delay_ms(1000);  // 1 sec delay ==> recommended

	PORTA = 0x81;
	_delay_ms(100);  // 0.1 sec delay
	PORTA = 0x42;
	_delay_ms(100);  // 0.1 sec delay
	PORTA = 0x24;
	_delay_ms(100);  // 0.1 sec delay
	PORTA = 0x18;
	_delay_ms(100);  // 0.1 sec delay
	PORTA = 0x18;
	_delay_ms(100);  // 0.1 sec delay
	PORTA = 0x24;
	_delay_ms(100);  // 0.1 sec delay
	PORTA = 0x42;
	_delay_ms(100);  // 0.1 sec delay
	PORTA = 0x81;
	_delay_ms(100);  // 0.1 sec delay
	PORTA = 0x00;

	for (;;)
		sleep_mode();

	return 0;
}
