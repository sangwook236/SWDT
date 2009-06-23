#include <avr/interrupt.h>
#include <avr/sfr_defs.h>


ISR(ADC_vect)
{
}

ISR(ANALOG_COMP_vect)
{
}

void adc_init()
{
	ADMUX = 0x00;
	// reference voltage source: uses internal 2.56V
	ADMUX |= _BV(REFS1);  // reference selection bit #1
	ADMUX |= _BV(REFS0);  // reference selection bit #0
	//ADMUX &= ~(_BV(REFS1));
	//ADMUX &= ~(_BV(REFS0));

	ADCSRA = 0x00;
	ADCSRA |= _BV(ADEN);  // ADC enable
	//ADCSRA |= _BV(ADSC);  // ADC start conversion
	ADCSRA &= ~(_BV(ADFR));  // ADC free running: off
	ADCSRA |= _BV(ADIE);  // ADC interrupt enable
	// ADC prescaler: 2╨паж
	ADCSRA &= ~(_BV(ADPS2));  // ADC prescaler select bit #2
	ADCSRA &= ~(_BV(ADPS1));  // ADC prescaler select bit #1
	ADCSRA &= ~(_BV(ADPS0));  // ADC prescaler select bit #0
}
