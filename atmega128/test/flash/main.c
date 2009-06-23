#include <avr/pgmspace.h>
#include <avr/interrupt.h>


void system_init()
{
	/*
	 *	analog comparator
	 */
	ACSR &= ~(_BV(ACIE));  // analog comparator interrupt disable
	ACSR |= _BV(ACD);  // analog comparator disable
}

int main(void)
{
	cli();
	system_init();
	sei();

	const prog_char ch = { 'Z' };
	const uint8_t val1 = pgm_read_byte(&ch);

	const prog_char ch_arr[] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'D', 'E', 'F', };
	const uint8_t val2 = pgm_read_byte(&ch_arr[3]);

	const prog_char *str1 = PSTR("test Flash ROM");
	const uint8_t val3_1 = pgm_read_byte(str1 + 5);
	const uint8_t val3_2 = pgm_read_byte(&str1[5]);

	//const prog_char str2[] PROGMEM = "test Flash ROM";
	const prog_char str2[] = "Flash ROM test";
	const uint8_t val4_1 = pgm_read_byte(str2 + 6);
	const uint8_t val4_2 = pgm_read_byte(&str2[6]);

	return 0;
}

