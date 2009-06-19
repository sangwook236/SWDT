#include <avr/sleep.h>
#include <avr/interrupt.h>
#include <util/delay.h>


void initSystem();

int main(void)
{
	void initPio();
	void initTimer();
	void initExtInt();

	cli();
	///EIFR = 0x00;
	initSystem();
	initPio();
	//initTimer();
	initExtInt();
	//EIFR = 0x00;
	sei();

	_delay_ms(1000);  // 1 sec delay ==> recommended

	PORTA = 0x81;
	_delay_ms(100);  // 0.1 sec delay
	PORTA = 0x42;
	_delay_ms(100);  // 0.1 sec delay
	PORTA = 0x24;
	_delay_ms(100);  // 0.1 sec delay
	PORTA = 0x18;
	_delay_ms(100);  // 0.1 sec delay
	PORTA = 0x18;
	_delay_ms(100);  // 0.1 sec delay
	PORTA = 0x24;
	_delay_ms(100);  // 0.1 sec delay
	PORTA = 0x42;
	_delay_ms(100);  // 0.1 sec delay
	PORTA = 0x81;
	_delay_ms(100);  // 0.1 sec delay
	PORTA = 0x00;

	for (;;)
		sleep_mode();

	return 0;
}

void initSystem()
{
	// Analog Comparator
	ACSR &= ~(_BV(ACIE));  // analog comparator interrupt disable
	ACSR |= _BV(ACD);  // analog comparator disable
}
