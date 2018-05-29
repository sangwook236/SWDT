#include "adc.h"
#include <avr/interrupt.h>
#include <avr/sfr_defs.h>


uint8_t mma7260_set(const uint8_t mode)
{
	// REFS1	REFS0		reference voltage source
	//--------------------------------------------------------------------
	// 0		0			use external AREF pin
	// 0		1			use external AVCC pin
	// 1		0			(reserved)
	// 1		1			uses internal 2.56V
	ADMUX &= ~(_BV(REFS1));
	ADMUX |= _BV(REFS0);

	// Analog channel and gain selection.
	if (0 == mode)  // X-axis.
	{
		ADMUX &= ~(_BV(MUX4));
		ADMUX &= ~(_BV(MUX3));
		ADMUX &= ~(_BV(MUX2));
		ADMUX &= ~(_BV(MUX1));
		ADMUX &= ~(_BV(MUX0));
	}
	else if (1 == mode)  // Y-axis.
	{
		ADMUX &= ~(_BV(MUX4));
		ADMUX &= ~(_BV(MUX3));
		ADMUX &= ~(_BV(MUX2));
		ADMUX |= _BV(MUX1);
		ADMUX &= ~(_BV(MUX0));
	}
	else if (2 == mode)  // Z-axis.
	{
		ADMUX &= ~(_BV(MUX4));
		ADMUX &= ~(_BV(MUX3));
		ADMUX |= _BV(MUX2);
		ADMUX &= ~(_BV(MUX1));
		ADMUX &= ~(_BV(MUX0));
	}
	else return 0;

	// ADC free running.
	ADCSRA &= ~(_BV(ADFR));  // If 1, ADC free running mode.

	// ADC prescaler select.
	// ADPS2	ADPS1	ADPS0		prescaler
	//--------------------------------------------------------------------
	// 0		0		0			2
	// 0		0		1			2
	// 0		1		0			4
	// 0		1		1			8
	// 1		0		0			16
	// 1		0		1			32
	// 1		1		0			64
	// 1		1		1			128
	ADCSRA |= _BV(ADEN);  // Prescaler can be set only when ADEN = 1.
		ADCSRA |= _BV(ADPS2);
		ADCSRA |= _BV(ADPS1);
		ADCSRA |= _BV(ADPS0);
	ADCSRA &= ~(_BV(ADEN));

	return 1;
}

