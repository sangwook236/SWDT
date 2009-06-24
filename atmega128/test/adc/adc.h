#if !defined(__SWL_AVR__ADC_H_)
#define __SWL_AVR__ADC_H_ 1


#include <avr/sfr_defs.h>


//-----------------------------------------------------------------------------
//

void adc_init();
void adc_start();
void adc_reset_complete();
uint8_t adc_is_complete();


#endif  // __SWL_AVR__ADC_H_
