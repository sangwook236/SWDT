#include "adc.h"
#include "usart.h"
#include <avr/sleep.h>
#include <avr/interrupt.h>
#include <util/delay.h>


void test_adc_basic();
void test_adc_mma7260();


void system_init()
{
 	/*
	 *	ADC
	 */
	set_sleep_mode(SLEEP_MODE_ADC);  // ADC noise reduction mode

 	/*
	 *	analog comparator
	 */
	ACSR &= ~(_BV(ACIE));  // analog comparator interrupt disable
	ACSR |= _BV(ACD);  // analog comparator disable

	/*
	 *	I/O port
	 */
	PORTA = 0x00;
	DDRA  = 0xFF;  // uses all pins on PortA for output
	PORTB = 0x00;
	DDRB  = 0x00;
	PORTC = 0x00;
	DDRC  = 0x00;
	PORTD = 0x00;
	DDRD  = 0x00;
	PORTE = 0x00;
	DDRE  = 0x00;
	PORTF = 0x00;
	DDRF  = 0xD0;
	PORTG = 0x00;
	DDRG  = 0x00;
}

int main(void)
{
	enum { MODE_ADC_BASIC = 0, MODE_ADC_MMA7260 };
	const uint8_t mode = MODE_ADC_MMA7260;

	cli();
	system_init();
	adc_init();
	usart0_init(57600UL);
	switch (mode)
	{
	case MODE_ADC_MMA7260:
		PORTF &= ~(_BV(PF5));
		PORTF &= ~(_BV(PF6));
		PORTF |= _BV(PF7);
		break;
	}
	sei();

	while (1)
	{
		switch (mode)
		{
		case MODE_ADC_BASIC:
			test_adc_basic(); 
			break;
		case MODE_ADC_MMA7260:
			test_adc_mma7260();
			break;
		}

		//sleep_mode();
	}

	return 0;
}

void test_adc_basic()
{
	uint8_t adc_set();

	adc_stop();
	adc_set();
	adc_start();

	while (!adc_is_complete()) ;

	const uint16_t val = adc_read_value();

	PORTA = val & 0xFF;

	//usart0_push_char(hex2ascii((val >> 12) & 0x0F));
	usart0_push_char(hex2ascii((val >> 8) & 0x03));
	usart0_push_char(hex2ascii((val >> 4) & 0x0F));
	usart0_push_char(hex2ascii(val & 0x0F));
	usart0_push_char(' ');

	_delay_ms(100);
}

void test_adc_mma7260()
{
	uint8_t mma7260_set(const uint8_t mode);

	const uint8_t XYZ = 0x0F;

	const uint16_t LOOP_COUNT = 16;
	const uint16_t LOOP_COUNT_BITS = 4;  // LOOP_COUNT = 2^LOOP_COUNT_BITS;

	uint32_t x_accel = 0, y_accel = 0, z_accel = 0;
	for (uint16_t i = 0; i < LOOP_COUNT; ++i)
	{
		// read X-axis
		if (0x01 & XYZ)
		{
			adc_stop();
			mma7260_set(0);
			adc_start();
			while (!adc_is_complete()) ;

			x_accel += adc_read_value();
		}

		// read Y-axis
		if (0x02 & XYZ)
		{
			adc_stop();
			mma7260_set(1);
			adc_start();
			while (!adc_is_complete()) ;

			y_accel += adc_read_value();
		}

		// read Z-axis
		if (0x04 & XYZ)
		{
			adc_stop();
			mma7260_set(2);
			adc_start();
			while (!adc_is_complete()) ;

			z_accel += adc_read_value();
		}
	}

	// read X-axis
	if (0x01 & XYZ)
	{
		//x_accel = (x_accel >> LOOP_COUNT_BITS);  // averaging: accel / 16
		x_accel = (x_accel >> LOOP_COUNT_BITS) & 0xFFFE;  // averaging

		PORTA = x_accel & 0xFF;
		if (usart0_is_empty())
		{
			usart0_push_char('x');
			//usart0_push_char(hex2ascii((x_accel >> 12) & 0x0F));
			usart0_push_char(hex2ascii((x_accel >> 8) & 0x03));
			usart0_push_char(hex2ascii((x_accel >> 4) & 0x0F));
			usart0_push_char(hex2ascii(x_accel & 0x0F));
			usart0_push_char(' ');
		}

		_delay_ms(100);
	}

	// read Y-axis
	if (0x02 & XYZ)
	{
		//y_accel = (y_accel >> LOOP_COUNT_BITS);  // averaging: accel / 16
		y_accel = (y_accel >> LOOP_COUNT_BITS) & 0xFFFE;  // averaging

		PORTA = y_accel & 0xFF;
		if (usart0_is_empty())
		{
			usart0_push_char('y');
			//usart0_push_char(hex2ascii((y_accel >> 12) & 0x0F));
			usart0_push_char(hex2ascii((y_accel >> 8) & 0x03));
			usart0_push_char(hex2ascii((y_accel >> 4) & 0x0F));
			usart0_push_char(hex2ascii(y_accel & 0x0F));
			usart0_push_char(' ');
		}
	
		_delay_ms(100);
	}

	// read Z-axis
	if (0x04 & XYZ)
	{
		//z_accel = (z_accel >> LOOP_COUNT_BITS);  // averaging: accel / 16
		z_accel = (z_accel >> LOOP_COUNT_BITS) & 0xFFFE;  // averaging

		PORTA = z_accel & 0xFF;
		if (usart0_is_empty())
		{
			usart0_push_char('z');
			//usart0_push_char(hex2ascii((z_accel >> 12) & 0x0F));
			usart0_push_char(hex2ascii((z_accel >> 8) & 0x03));
			usart0_push_char(hex2ascii((z_accel >> 4) & 0x0F));
			usart0_push_char(hex2ascii(z_accel & 0x0F));
			usart0_push_char(' ');
		}

		_delay_ms(100);
	}
}
