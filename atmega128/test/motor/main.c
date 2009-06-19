#include <avr/sleep.h>
#include <avr/interrupt.h>


int main(void)
{
	void initAdc();
	void initAnalogComparator();
	void initPio();
	void initTimer();
	void initExtInt();

	cli();
	initAdc();
	initAnalogComparator();
	initPio();
	initTimer();
	initExtInt();
	sei();

	for (;;)
		sleep_mode();

	return 0;
}
