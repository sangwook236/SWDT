#include <avr/io.h>
#include <util/delay.h>

// ICC-AVR application builder : 2007-06-27 ¿ÀÈÄ 5:20:51.
//	Target : M128.
//	Crystal: 16.000Mhz.
/*
#include <iom128v.h>
#include <macros.h>
#include <KTM128.h>

void port_init(void)
{
	PORTA = 0x00;
	DDRA  = 0x00;
	PORTB = 0x00;
	DDRB  = 0x00;
	PORTC = 0x00; //m103 output only
	DDRC  = 0x00;
	PORTD = 0x00;
	DDRD  = 0x00;
	PORTE = 0x00;
	DDRE  = 0x00;
	PORTF = 0x00;
	DDRF  = 0x00;
	PORTG = 0x00;
	DDRG  = 0x00;
}

// Call this routine to initialize all peripherals.
void init_devices(void)
{
	// Stop errant interrupts until set up.
	CLI();  // Cisable all interrupts.
	XDIV  = 0x00;  // xtal divider.
	XMCRA = 0x00;  // External memory.
	port_init();

	MCUCR = 0x80;
	EICRA = 0x00;  // Extended ext ints.
	EICRB = 0x00;  // Extended ext ints.
	EIMSK = 0x00;
	TIMSK = 0x00;  // Timer interrupt sources.
	ETIMSK = 0x00;  // Extended timer interrupt sources.
	SEI();  // Re-enable interrupts.
	// All peripherals are now initialized.
}

void main()
{
	int i;
	// Cathode.
	unsigned int arr[] = {0x3f,0x06,0x5b,0x4f,0x66,0x6d,0x7d,0x27,0x7f,0x6f,0x77,0x7c,0x39,0x5e,0x79,0x71};

	init_devices();
	while(1)
	{
		DDRB = 0xff;
		DDRD = 0xff;

		for (i = 0; i < 16; ++i)
		{
			PORTB = ~0xff;
			PORTD = arr[i];
			delay(500);
		}
	}
}
*/

void displayCathodeCommon()
{
	const uint8_t str_7segment_cathode[] = {
		0x3f, 0x06, 0x5b, 0x4f, 0x66, 0x6d, 0x7d, 0x27,
		0x7f, 0x6f, 0x77, 0x7c, 0x39, 0x5e, 0x79, 0x71
	};

	uint8_t display_num[4] = { 0, };
	uint16_t i = 0;
	while (1)
	{
        display_num[0] = (i % 10000) / 1000;
        display_num[1] = (i % 1000) / 100;
        display_num[2] = (i % 100) / 10;
        display_num[3] = (i % 10);

		for (int j = 0; j < 4; ++j)
		{
			PORTC = 0x0F;
			PORTA = str_7segment_cathode[display_num[j]];
			PORTC = ~(0x01 << j);

			// A maximal possible delay time is (262.14 / Fosc in MHz) ms.
			// If Fosc = 16 MHz, a maximal possible delay time = 16.38375 ms.
			//	500 ms -> 10 ms * 50 count.
			//for (int k = 0; k < 5; ++k)
				_delay_ms(5);
		}

		++i;
	}
}

void four_digit_seven_segment_cathode_commmon(const uint16_t fourDigits)
{
	const uint8_t str_7segment_cathode[] =
	{
		0x3f, 0x06, 0x5b, 0x4f, 0x66, 0x6d, 0x7d, 0x27,
		0x7f, 0x6f, 0x77, 0x7c, 0x39, 0x5e, 0x79, 0x71
	};

	const uint8_t display_num[4] = 
	{
    	(fourDigits % 10000) / 1000,
    	(fourDigits % 1000) / 100,
    	(fourDigits % 100) / 10,
    	(fourDigits % 10)
	};

	for (int j = 0; j < 4; ++j)
	{
		PORTC = 0x0F;
		PORTA = str_7segment_cathode[display_num[j]];
		//if (3 == j) PORTA += 0x80;  // the last DP(dot point) is displayed
		PORTC = ~(0x01 << j);

		// A maximal possible delay time is (262.14 / Fosc in MHz) ms.
		// If Fosc = 16 MHz, a maximal possible delay time = 16.38375 ms.
		//	500 ms -> 10 ms * 50 count
		//for (int k = 0; k < 5; ++k)
			_delay_ms(5);
	}
}
