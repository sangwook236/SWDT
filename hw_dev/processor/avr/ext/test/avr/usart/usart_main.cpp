#include "usart.h"
#include <avr/sleep.h>
#include <avr/interrupt.h>
#include <util/delay.h>


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
	// It makes port input register(PINn) high-impedance state that port output register(PORTn) outputs 0(low)
	// so that we can share the pin with other devices.
	//PORTA = 0x00;

	// Uses all pins on PortD for output.
	DDRD = 0xFF;
*/
	DDRA = 0xFF;  // Uses all pins on PortA for output.
	DDRD = 0xFF;  // Uses all pins on PortD for output.
}

}  // namespace local
}  // unnamed namespace

namespace my_usart {
}  // namespace my_usart

int usart_main(int argc, char *argv[])
{
	cli();
	local::system_init();
	usart0_init(57600UL);
	usart1_init(57600UL);
	sei();

	PORTD = 0xFF;
	_delay_ms(100);
	PORTD = 0x00;

	while (1)
	{
		if (!usart0_is_empty())
		{
/*
#if 0
			uint8_t hex = usart0_top_char();
			usart0_pop_char();
#else
			uint8_t ascii = usart0_top_char();
			usart0_pop_char();
			uint8_t hex = ascii2hex(ascii);
#endif
			uint16_t num = 0xABCD;
			usart1_push_char(hex2ascii((num >> 12) & 0x0F));
			usart1_push_char(hex2ascii((num >> 8) & 0x0F));
			usart1_push_char(hex2ascii((num >> 4) & 0x0F));
			usart1_push_char(hex2ascii(num & 0x0F));
*/
			uint8_t ascii = usart0_top_char();
			usart0_pop_char();
			uint8_t hex = ascii2hex(ascii);

			PORTA = hex;

			usart0_push_char(ascii);
		}

		if (!usart1_is_empty())
		{
/*
#if 0
			uint8_t hex = usart1_top_char();
			usart1_pop_char();
#else
			uint8_t ascii = usart1_top_char();
			usart1_pop_char();
			uint8_t hex = ascii2hex(ascii);
#endif
			uint16_t num = 0xABCD;
			usart1_push_char(hex2ascii((num >> 12) & 0x0F));
			usart1_push_char(hex2ascii((num >> 8) & 0x0F));
			usart1_push_char(hex2ascii((num >> 4) & 0x0F));
			usart1_push_char(hex2ascii(num & 0x0F));

*/
			uint8_t ascii = usart1_top_char();
			usart1_pop_char();
			uint8_t hex = ascii2hex(ascii);

			PORTA = hex;

			usart1_push_char(ascii);
		}

		//sleep_mode();
	}

	return 0;
}
