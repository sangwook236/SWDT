#include "usart.h"


uint8_t hex2ascii(const uint8_t hex)
{
	if (0x00 <= (int8_t)hex && (int8_t)hex <= 0x09)
		return hex + '0';
	//else if (0x0a <= hex && hex <= 0x0f)
	else if (0x0A <= hex && hex <= 0x0F)
		//return hex - 0x0A + (doesConvToUpperCase ? 'A' : 'a');
		return hex - 0x0A + 'A';
	else return (uint8_t)-1;
}

uint8_t ascii2hex(const uint8_t ascii)
{
	if ('0' <= ascii && ascii <= '9')
		return ascii - '0';
	else if ('a' <= ascii && ascii <= 'f')
		return ascii - 'a' + 10;
	else if ('A' <= ascii && ascii <= 'F')
		return ascii - 'A' + 10;
	else return (uint8_t)-1;
}
