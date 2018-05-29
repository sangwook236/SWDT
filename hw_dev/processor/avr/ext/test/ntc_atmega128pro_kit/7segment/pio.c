#include <avr/io.h>


void initPio()
{
	// Uses all pins on PortA & PortC for output.
	DDRA = 0xFF;
	DDRC = 0xFF;
}
