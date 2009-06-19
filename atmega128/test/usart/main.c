#include <avr/sleep.h>
#include <avr/interrupt.h>
#include <util/delay.h>


void initSystem();

int main(void)
{
	void initPio();
	void initUsart();

	int8_t isEmpty_Usart0();
	int8_t pushChar_Usart0(const uint8_t ch);
	void popChar_Usart0();
	uint8_t topChar_Usart0();
	uint8_t hex2ascii(const uint8_t hex);
	uint8_t ascii2hex(const uint8_t ascii);

	cli();
	initSystem();
	initPio();
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

void initSystem()
{
	// Analog Comparator
	ACSR &= ~(_BV(ACIE));  // analog comparator interrupt disable
	ACSR |= _BV(ACD);  // analog comparator disable
}
