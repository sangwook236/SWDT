#include <avr/io.h>
#include <avr/interrupt.h>
#include <util/delay.h>

#define TargetPort PORTA

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
   	//outp(0x00,DDRA);
	DDRA = 0x00;
	// it makes port input register(PINn) internally pulled-up state that port output register(PORTn) outputs 1(high)
	PORTA = 0xFF;
	// it makes port input register(PINn) high-impedence state that port output register(PORTn) outputs 0(low)
	// so that we can share the pin with other devices
	//PORTA = 0x00;

	// uses all pins on PortD for output
   	//outp(0xFF,DDRD);
	DDRD = 0xFF;
*/
	DDRA = 0xFF;  // uses all pins on PortA for output
}

int main(void)
{
  	uint8_t led;

	cli();
	system_init();
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
