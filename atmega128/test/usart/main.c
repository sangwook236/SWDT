#include "usart.h"
#include <avr/sleep.h>
#include <avr/interrupt.h>
#include <util/delay.h>


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

	// uses all pins on PortD for output
	DDRD = 0xFF;
*/
	DDRA = 0xFF;  // uses all pins on PortA for output
	DDRD = 0xFF;  // uses all pins on PortD for output
}

int main(void)
{
	cli();
	system_init();
	usart0_init();
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
			usart0_push_char(hex2ascii((hex + 1) % 256));

			PORTD = hex;
*/
			uint8_t ascii = usart0_top_char();
			usart0_pop_char();
			uint8_t hex = ascii2hex(ascii);

			PORTA = hex;

			uint16_t num = 0xABCD;
			usart0_push_char(hex2ascii((num >> 12) & 0x0F));
			usart0_push_char(hex2ascii((num >> 8) & 0x0F));
			usart0_push_char(hex2ascii((num >> 4) & 0x0F));
			usart0_push_char(hex2ascii(num & 0x0F));
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
			usart1_push_char(hex2ascii((hex + 1) % 256));

			PORTD = hex;
*/
			uint8_t ascii = usart1_top_char();
			usart1_pop_char();
			uint8_t hex = ascii2hex(ascii);

			PORTA = hex;

			uint16_t num = 0xABCD;
			usart1_push_char(hex2ascii((num >> 12) & 0x0F));
			usart1_push_char(hex2ascii((num >> 8) & 0x0F));
			usart1_push_char(hex2ascii((num >> 4) & 0x0F));
			usart1_push_char(hex2ascii(num & 0x0F));
		}

		//sleep_mode();
	}

	return 0;
}
