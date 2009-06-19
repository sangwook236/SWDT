#include <avr/sleep.h>
#include <avr/interrupt.h>
#include <util/delay.h>


void initSystem();


int main(void)
{
	void initPio();
	void initAdc();
	void startAdc();
	void resetAdcComplete();
	uint8_t isAdcComplete();
	void initAnalogComparator();
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
	initAdc();
	initAnalogComparator();
	initUsart();
	sei();

	for (;;)
	{
		//sleep_mode();
		startAdc();

		while (!isAdcComplete());

		uint16_t result = ADCL;
		result |= (ADCH << 8);

		PORTA = (result & 0xFF);

		//pushChar_Usart0(hex2ascii((result >> 12) & 0x0F));
		pushChar_Usart0(hex2ascii((result >> 8) & 0x03));
		pushChar_Usart0(hex2ascii((result >> 4) & 0x0F));
		pushChar_Usart0(hex2ascii(result & 0x0F));
		pushChar_Usart0(' ');

		resetAdcComplete();

		// a maximal possible delay time is (262.14 / Fosc in MHz) ms
		// if Fosc = 16 MHz, a maximal possible delay time = 16.38375 ms
		//	100 ms -> 10 ms * 50 count
		for (int i = 0; i < 10; ++i)
			_delay_ms(10);
	}

	return 0;
}

void initSystem()
{
	// ADC Noise Reduction Mode
	set_sleep_mode(SLEEP_MODE_ADC);
}
