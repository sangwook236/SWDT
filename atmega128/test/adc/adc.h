#if !defined(__SWL_AVR__ADC_H_)
#define __SWL_AVR__ADC_H_ 1


#include <avr/sfr_defs.h>


//#define __SWL_AVR__USE_ADC_INTERRUPT 1


//-----------------------------------------------------------------------------
//

void adc_init();
void adc_start();
void adc_stop();
uint8_t adc_is_complete();
uint16_t adc_read_value();


#endif  // __SWL_AVR__ADC_H_
