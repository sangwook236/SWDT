#include <avr/io.h>
#include <avr/interrupt.h>
#include <stdint.h>


void initSystem();

int main()
{
	void initPio();
	void four_digit_seven_segment_anode_commmon(const uint16_t four_digits);
	void four_digit_seven_segment_cathode_commmon(const uint16_t four_digits);

	cli();
	initSystem();
	initPio();
	sei();

	const uint16_t num1 = 1234;
	const uint16_t num2 = 5678;
	uint16_t num = num1;
	uint16_t i = 0;
	while (1)
	{
		//four_digit_seven_segment_anode_commmon(num);
		four_digit_seven_segment_cathode_commmon(num);

		i = (i + 1) % 100;
		if (0 == i)
			num = (num1 == num) ? num2 : num1;
	}

	return 0;
}

void initSystem()
{
	// Analog Comparator
	ACSR &= ~(_BV(ACIE));  // analog comparator interrupt disable
	ACSR |= _BV(ACD);  // analog comparator disable
}
