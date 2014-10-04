#include "adc.h"
#include <avr/interrupt.h>
#include <avr/sfr_defs.h>


#if defined(__SWL_AVR__USE_ADC_INTERRUPT)

static uint8_t gs_isAdcComplete = 0;

ISR(ADC_vect)
{
	gs_isAdcComplete = 1;
}

#endif  // __SWL_AVR__USE_ADC_INTERRUPT

static const uint8_t isLeftAdjustment = 0;

void adc_init()
{
	// ADC left adjust result
	if (isLeftAdjustment)
		ADMUX |= _BV(ADLAR);  // if 1, use left adjustment
	else
		ADMUX &= ~(_BV(ADLAR));  // if 0, use right adjustment.

	// FIXME [check] >>
#if defined(__SWL_AVR__USE_ADC_INTERRUPT)
	ADCSRA |= _BV(ADIE);  // ADC interrupt enable
#else
#endif  // __SWL_AVR__USE_ADC_INTERRUPT
}

uint8_t adc_set()
{
	// REFS1	REFS0		reference voltage source
	//--------------------------------------------------------------------
	// 0		0			use external AREF pin
	// 0		1			use external AVCC pin
	// 1		0			(reserved)
	// 1		1			uses internal 2.56V
	ADMUX |= _BV(REFS1);
	ADMUX |= _BV(REFS0);

	// analog channel and gain selection
	ADMUX &= ~(_BV(MUX4));
	ADMUX &= ~(_BV(MUX3));
	ADMUX &= ~(_BV(MUX2));
	ADMUX &= ~(_BV(MUX1));
	ADMUX &= ~(_BV(MUX0));

	// ADC free running
	ADCSRA &= ~(_BV(ADFR));  // if 1, ADC free running mode

	// ADC prescaler select
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
	ADCSRA |= _BV(ADEN);  // prescaler can be set only when ADEN = 1
		ADCSRA |= _BV(ADPS2);
		ADCSRA |= _BV(ADPS1);
		ADCSRA |= _BV(ADPS0);
	ADCSRA &= ~(_BV(ADEN));

	return 1;
}

void adc_start()
{
#if defined(__SWL_AVR__USE_ADC_INTERRUPT)
	gs_isAdcComplete = 0;
#else
#endif  // __SWL_AVR__USE_ADC_INTERRUPT

	ADCSRA |= _BV(ADEN) | _BV(ADSC);  // ADC enable & ADC start conversion
}

void adc_stop()
{
	ADCSRA &= ~(_BV(ADEN));  // ADC disable
}

uint8_t adc_is_complete()
{
#if defined(__SWL_AVR__USE_ADC_INTERRUPT)
	return 1 == gs_isAdcComplete;
#else
	return bit_is_set(ADCSRA, ADIF);
#endif  // __SWL_AVR__USE_ADC_INTERRUPT
}

uint16_t adc_read_value()
{
	// FIXME [check] >>
	if (isLeftAdjustment)
	{
		const uint8_t low = ADCL;
		const uint8_t high = ADCH;
		return (((high << 8) | low) >> 6) & 0x03FF;
	}
	else
	{
		const uint8_t low = ADCL;
		const uint8_t high = ADCH;
		return ((high << 8) | low) & 0x03FF;;
	}
}
