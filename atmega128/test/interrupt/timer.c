#include <avr/interrupt.h>
#include <avr/sfr_defs.h>


ISR(TIMER0_OVF_vect)
{
	static uint8_t led = 0x01;
	static uint16_t tick_count = 0;

	// 0.1msec timer
	// 0.1msec * 10000 = 1sec
	if (10000 == ++tick_count) {
		PORTD = led;
		led <<= 1;  // moves to next LED
		if (!led)  // overflow: starts with Port D0 again
			led = 0x01;

		tick_count = 0;
	}

	// 2MHz -> 0.5usec. 200 counts = 200 * 0.5usec = 0.1msec
	// 255(FFh) - 200(C8h) = 55(37h)
	TCNT0 = 0x37;
}

ISR(TIMER1_OVF_vect)
{
	static uint8_t led = 0x01;

	PORTD = led;
	led <<= 1;  // moves to next LED
	if (!led)  // overflow: starts with Port D0 again
		led = 0x01;

	// 16KHz = 16384 pulses/sec -> 16384 = 4000h: 65535(FFFFh) - 16384(4000h) = 49151(BFFFh)
	TCNT1 = 0xBFFF;
	//TCNT1H = 0xBF;
	//TCNT1L = 0xFF;
}

void initTimer()
{
	TCCR0 = 0x02;  // normal mode(up counter), Fcpu / 8 = 16MHz / 8 = 2MHz
	TCCR1A = 0x00;
	TCCR1B = 0x05;  // normal mode(up counter), Fcpu / 1024 = 16MHz / 1024 = 16KHz
	TCCR1C = 0x00;

	// 2MHz -> 0.5usec. 200 counts = 200 * 0.5usec = 0.1msec
	// 255(FFh) - 200(C8h) = 55(37h)
	TCNT0 = 0x37;

	// 16KHz = 16384 pulses/sec -> 16384 = 4000h: 65535(FFFFh) - 16384(4000h) = 49151(BFFFh)
	TCNT1 = 0xBFFF;
	//TCNT1H = 0xBF;
	//TCNT1L = 0xFF;

	TIMSK = 0x00;
	TIMSK &= ~(_BV(TOIE0));  // disables Timer0 overflow interrupt
	TIMSK |= _BV(TOIE1);  // enables Timer1 overflow interrupt
	ETIMSK = 0x00;
}
