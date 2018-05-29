#include <avr/io.h>
#include <util/delay.h>


void four_digit_seven_segment_anode_commmon(const uint16_t four_digits)
{
	const uint8_t str_7segment_cathode[] =
	{
		0x3f, 0x06, 0x5b, 0x4f, 0x66, 0x6d, 0x7d, 0x27,
		0x7f, 0x6f, 0x77, 0x7c, 0x39, 0x5e, 0x79, 0x71
	};

	const uint8_t display_num[4] = 
	{
    	(uint8_t)((four_digits % 10000) / 1000),
    	(uint8_t)((four_digits % 1000) / 100),
    	(uint8_t)((four_digits % 100) / 10),
    	(uint8_t)(four_digits % 10)
	};

	for (int j = 0; j < 4; ++j)
	{
		PORTC = 0x0F;
		PORTA = str_7segment_cathode[display_num[j]];
		//if (3 == j)  PORTA += 0x80;  // The last DP(dot point) is displayed.
		PORTC = ~(0x01 << j);

		// A maximal possible delay time is (262.14 / Fosc in MHz) ms.
		// If Fosc = 16 MHz, a maximal possible delay time = 16.38375 ms.
		//	500 ms -> 10 ms * 50 count.
		//for (int k = 0; k < 5; ++k)
		_delay_ms(5);
	}
}

void four_digit_seven_segment_cathode_commmon(const uint16_t four_digits)
{
	const uint8_t str_7segment_cathode[] =
	{
		0x3f, 0x06, 0x5b, 0x4f, 0x66, 0x6d, 0x7d, 0x27,
		0x7f, 0x6f, 0x77, 0x7c, 0x39, 0x5e, 0x79, 0x71
	};

	const uint8_t display_num[4] = 
	{
    	(uint8_t)((four_digits % 10000) / 1000),
    	(uint8_t)((four_digits % 1000) / 100),
    	(uint8_t)((four_digits % 100) / 10),
    	(uint8_t)(four_digits % 10)
	};

	for (int j = 0; j < 4; ++j)
	{
		PORTC = 0x0F;
		PORTA = str_7segment_cathode[display_num[j]];
		//if (3 == j)  PORTA += 0x80;  // The last DP(dot point) is displayed.
		PORTC = ~(0x01 << j);

		// A maximal possible delay time is (262.14 / Fosc in MHz) ms.
		// If Fosc = 16 MHz, a maximal possible delay time = 16.38375 ms.
		//	500 ms -> 10 ms * 50 count.
		//for (int k = 0; k < 5; ++k)
		_delay_ms(5);
	}
}
