#include <avr/interrupt.h>
#include <avr/sfr_defs.h>


ISR(TIMER0_OVF_vect)
{
/*
	static uint8_t led = 0x01;
	static uint16_t tick_count = 0;

	// 0.1msec timer.
	// 0.1msec * 10000 = 1sec.
	if (10000 == ++tick_count) {
		PORTD = led;
		led <<= 1;  // Moves to next LED.
		if (!led)  // Overflow: Starts with Port D0 again.
			led = 0x01;

		tick_count = 0;
	}
*/
}

ISR(TIMER0_COMP_vect)
{
	static uint8_t led = 0x01;
	static uint16_t tick_count = 0;

	// 0.1msec timer.
	// 0.1msec * 10000 = 1sec.
	if (10000 == ++tick_count) {
		PORTD = led;
		led <<= 1;  // Moves to next LED.
		if (!led)  // Overflow: Starts with Port D0 again.
			led = 0x01;

		tick_count = 0;
	}
}
/*
ISR(TIMER1_OVF_vect)
{
	static uint8_t led = 0x01;

	PORTD = led;
	led <<= 1;  // Moves to next LED.
	if (!led)  // Overflow: Starts with Port D0 again.
		led = 0x01;

	// 16KHz = 16384 pulses/sec -> 16384 = 4000h: 65535(FFFFh) - 16384(4000h) = 49151(BFFFh).
	TCNT1 = 0xBFFF;
	//TCNT1H = 0xBF;
	//TCNT1L = 0xFF;
}

ISR(TIMER1_COMPA_vect)
{
}

ISR(TIMER1_COMPB_vect)
{
}

ISR(TIMER1_COMPC_vect)
{
}

ISR(TIMER1_CAPT_vect)
{
}
*/

void timer_init()
{
	TCCR0 = 0x6A;  // Fast PWM mode(up counter), normal PWM output, Fcpu / 8 = 16MHz / 8 = 2MHz.
	TCCR1A = 0x00;
	TCCR1B = 0x05;  // Normal mode(up counter), Fcpu / 1024 = 16MHz / 1024 = 16KHz.
	TCCR1C = 0x00;

	//
	TCNT0 = 0x00;
	OCR0 = 192;  // Duty ratio: (128 + 1) / 256 = ~50%, (192 + 1) / 256 = ~75%.

	// 16KHz = 16384 pulses/sec -> 16384 = 4000h: 65535(FFFFh) - 16384(4000h) = 49151(BFFFh).
	TCNT1 = 0xBFFF;
	//TCNT1H = 0xBF;
	//TCNT1L = 0xFF;
	OCR1A = 0x1000;
	//OCR1AH = 0x10;
	//OCR1AL = 0x00;
	OCR1B = 0x1000;
	//OCR1BH = 0x10;
	//OCR1BL = 0x00;
	OCR1C = 0x1000;
	//OCR1CH = 0x10;
	//OCR1CL = 0x00;

	TIMSK = 0x00;
	TIMSK |= _BV(TOIE0);  // Enables Timer0 overflow interrupt.
	TIMSK |= _BV(OCIE0);  // Enables Timer0 output compare match interrupt.
	TIMSK &= ~(_BV(TOIE1));  // Disables Timer1 overflow interrupt.
	ETIMSK = 0x00;
}
