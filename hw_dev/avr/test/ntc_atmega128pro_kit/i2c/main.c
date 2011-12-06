#include <avr/sleep.h>
#include <avr/interrupt.h>


void initSystem();

int main(void)
{
	void initPio();
	void initI2c();

	cli();
	initSystem();
	initPio();
	initI2c();
	sei();

	for (;;) {
		sleep_mode(); 
	}

	return 0;
}

void initSystem()
{
	// Analog Comparator
	ACSR &= ~(_BV(ACIE));  // analog comparator interrupt disable
	ACSR |= _BV(ACD);  // analog comparator disable
}
