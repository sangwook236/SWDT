#if !defined(__SWL_AVR__USART_H_)
#define __SWL_AVR__USART_H_ 1


#include <avr/sfr_defs.h>


//#define __USE_CLASS_IN_USART 1


//-----------------------------------------------------------------------------
//

uint8_t hex2ascii(const uint8_t hex);
uint8_t ascii2hex(const uint8_t ascii);

//-----------------------------------------------------------------------------
//

void usart0_init(const uint32_t baudrate);
void usart0_init_buffer();
int8_t usart0_push_char(const uint8_t ch);
void usart0_pop_char();
uint8_t usart0_top_char();
int8_t usart0_is_empty();
uint32_t usart0_get_size();

//-----------------------------------------------------------------------------
//

void usart1_init(const uint32_t baudrate);
void usart1_init_buffer();
int8_t usart1_push_char(const uint8_t ch);
void usart1_pop_char();
uint8_t usart1_top_char();
int8_t usart1_is_empty();
uint32_t usart1_get_size();


#endif  // __SWL_AVR__USART_H_
