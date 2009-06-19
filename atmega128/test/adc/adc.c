#include <avr/interrupt.h>
#include <avr/sfr_defs.h>


static uint8_t gs_isAdcComplete = 0;

ISR(ADC_vect)
{
	gs_isAdcComplete = 1;
}

ISR(ANALOG_COMP_vect)
{
}

void initAdc()
{
	ADMUX = 0x00;
	// reference voltage source: uses internal 2.56V
	ADMUX |= _BV(REFS1);  // reference selection bit #1
	ADMUX |= _BV(REFS0);  // reference selection bit #0
	//ADMUX &= ~(_BV(REFS1));
	//ADMUX &= ~(_BV(REFS0));
	//ADMUX |= _BV(ADLAR);  // ADC left adjust result
	// analog channel and gain selection
	//ADMUX |= _BV(MUX3);  // analog channel and gain selection bit #3
	//ADMUX |= _BV(MUX2);  // analog channel and gain selection bit #2
	//ADMUX |= _BV(MUX1);  // analog channel and gain selection bit #1
	//ADMUX |= _BV(MUX0);  // analog channel and gain selection bit #0

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

void startAdc()
{
	gs_isAdcComplete = 0;
	ADCSRA |= _BV(ADSC);  // ADC start conversion
}

void resetAdcComplete()
{
	gs_isAdcComplete = 0;
}

uint8_t isAdcComplete()
{
	return 1 == gs_isAdcComplete;
}

void initAnalogComparator()
{
	ACSR &= ~(_BV(ACIE));  // analog comparator interrupt disable
	ACSR |= _BV(ACD);  // analog comparator disable
}
