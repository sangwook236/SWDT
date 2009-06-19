#include <avr/io.h>


void initPio()
{
 	DDRA = 0xFF;  // uses all pins on PortA for output
}
