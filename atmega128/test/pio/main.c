#include <avr/io.h>
#include <avr/interrupt.h>
#include <util/delay.h>

#define TargetPort PORTA

void initSystem();

int main(void)
{
	void initPio();

  	uint8_t led;

	cli();
	initSystem();
	initPio();
	sei();

	led = 1;  // init variable representing the LED state

	for (;;) {
		//outp(led, TargetPort);  // invert the output since a zero means: LED on
		TargetPort = led;
		led <<= 1;  // move to next LED
		if (!led)  // overflow: start with Port C0 again
			led = 1;
/*
		for (int i = 0; i < 30000; ++i)  // outer delay loop
			for(int j = 0; j < 30000; ++j)  // inner delay loop
				++k;  // just do something - could also be a NOP
*/
		// a maximal possible delay time is (262.14 / Fosc in MHz) ms
		// if Fosc = 16 MHz, a maximal possible delay time = 16.38375 ms
		//	500 ms -> 10 ms * 50 count
		for (int i = 0; i < 50; ++i)
			_delay_ms(10);
	}

	return 0;
}

void initSystem()
{
	// Analog Comparator
	ACSR &= ~(_BV(ACIE));  // analog comparator interrupt disable
	ACSR |= _BV(ACD);  // analog comparator disable
}
