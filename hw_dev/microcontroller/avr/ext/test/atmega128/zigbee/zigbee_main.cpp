#include "../usart/usart.h"
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

	// uses all pins on PortD for output
	DDRD = 0xFF;
*/
	DDRA = 0xFF;  // uses all pins on PortA for output
	// for two yellow LEDs
	DDRG = 0x03;
}

}  // namespace local
}  // unnamed namespace

namespace my_zigbee {
}  // namespace my_zigbee

int zigbee_main(void)
{
	cli();
	local::system_init();
	usart0_init(9600UL);  // 9600 bps. for connecting to ZigBee.
	usart1_init(57600UL);  // 57600 bps. for connecting to PC
	sei();

	PORTA = 0xFF;
	_delay_ms(500);
	PORTA = 0x00;

	uint8_t flag = 0;
	while (1)
	{
		// (???) -> ZigBee ..... ZigBee -> ATmega128 -> PC
		if (!usart0_is_empty())
		{
			const uint8_t ascii = usart0_top_char();
			usart0_pop_char();

			PORTA = ascii2hex(ascii);

			usart1_push_char(ascii);

			if (0 == flag)
			{
				PORTG =  0x01;
				flag = 1;
			}
			else
			{
				PORTG =  0x02;
				flag = 0;
			}
		}

		// PC -> ATmega128 -> ZigBee ..... ZigBee -> (???)
		if (!usart1_is_empty())
		{
			const uint8_t ascii = usart1_top_char();
			usart1_pop_char();

			PORTA = ascii2hex(ascii);

			usart0_push_char(ascii);

			if (0 == flag)
			{
				PORTG =  0x01;
				flag = 1;
			}
			else
			{
				PORTG =  0x02;
				flag = 0;
			}
		}

		// sending rate
		//	XBee,XBee-PRO: >= 200ms
		//	XBee2: <= 3s
		//_delay_ms(500);

		//sleep_mode();
	}

	return 0;
}
