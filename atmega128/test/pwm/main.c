#include <avr/sleep.h>
#include <avr/interrupt.h>


void initSystem();

int main(void)
{
	void initPio();
	void initTimer();

	cli();
	initSystem();
	initPio();
	initTimer();
	sei();

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
