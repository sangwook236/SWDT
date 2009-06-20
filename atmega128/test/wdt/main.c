#include <avr/wdt.h>
#include <avr/io.h>
#include <avr/interrupt.h>
#include <util/delay.h>


void initSystem()
{
	/*
	 *	watchdog timer
	 */
	wdt_enable(WDTO_2S);

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
}

int main(void)
{
	const int8_t doesUseWdt = 0;
  	uint8_t led;

	cli();
	initSystem();
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
