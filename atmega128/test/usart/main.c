#include <avr/sleep.h>
#include <avr/interrupt.h>
#include <util/delay.h>


void initSystem()
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
	void initUsart();

	int8_t isEmpty_Usart0();
	int8_t pushChar_Usart0(const uint8_t ch);
	void popChar_Usart0();
	uint8_t topChar_Usart0();
	uint8_t hex2ascii(const uint8_t hex);
	uint8_t ascii2hex(const uint8_t ascii);

	cli();
	initSystem();
	initUsart();
	sei();

	PORTD = 0xFF;
	_delay_ms(100);
	PORTD = 0x00;

	for (;;) {
		if (!isEmpty_Usart0()) {
/*
#if 0
			uint8_t hex = topChar_Usart0();
			popChar_Usart0();
#else
			uint8_t ascii = topChar_Usart0();
			popChar_Usart0();
			uint8_t hex = ascii2hex(ascii);
#endif
			pushChar_Usart0(hex2ascii((hex + 1) % 256));

			PORTD = hex;
*/
			uint8_t ascii = topChar_Usart0();
			popChar_Usart0();
			uint8_t hex = ascii2hex(ascii);

			PORTA = hex;

			uint16_t num = 0xABCD;
			pushChar_Usart0(hex2ascii((num >> 12) & 0x0F));
			pushChar_Usart0(hex2ascii((num >> 8) & 0x0F));
			pushChar_Usart0(hex2ascii((num >> 4) & 0x0F));
			pushChar_Usart0(hex2ascii(num & 0x0F));
		}

		//sleep_mode();
	}

	return 0;
}
