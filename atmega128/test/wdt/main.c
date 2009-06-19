#include <avr/wdt.h>
#include <avr/io.h>
#include <avr/interrupt.h>
#include <util/delay.h>


void initSystem();

int main(void)
{
	void initPio();

	const int8_t doesUseWdt = 0;
  	uint8_t led;

	cli();
	initSystem();
	initPio();
	sei();

	led = 1;  // init variable representing the LED state

	for (;;) {
		PORTD = led;  // invert the output since a zero means: LED on
		led <<= 1;  // move to next LED
		if (!led)  // overflow: start with Port C0 again
			led = 1;

		for (int i = 0; i < 50; ++i)
			_delay_ms(10);  // the maximal possible delay is 262.14 ms / F_CPU in MHz
							// if F_CPU = 16MHz, maximal possible delay = 16.38375
							// 10 ms * 50 count = 500 ms


		if (doesUseWdt)
			wdt_reset();
	}

	return 0;
}

void initSystem()
{
	// Watchdog Timer
	wdt_enable(WDTO_2S);

	// Analog Comparator
	ACSR &= ~(_BV(ACIE));  // analog comparator interrupt disable
	ACSR |= _BV(ACD);  // analog comparator disable
}
