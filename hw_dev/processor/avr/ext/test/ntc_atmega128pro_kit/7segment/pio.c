#include <avr/io.h>


void initPio()
{
	// uses all pins on PortA & PortC for output
	DDRA = 0xFF;
	DDRC = 0xFF;
}
