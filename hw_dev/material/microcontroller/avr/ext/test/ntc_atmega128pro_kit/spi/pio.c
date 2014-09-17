#include <avr/io.h>


void initPio()
{
	// uses all pins on PortA & PortC and PortB4 pin as output pins
	// uses all pins on PortB except B4 as input pins
	DDRA = 0xFF;
	//DDRB = 0x10;
	DDRB = _BV(PB4);
	DDRC = 0xFF;
}
