#include <avr/io.h>


void initPio()
{
/*
	// uses all pins on PortA for input
	DDRA = 0x00;
	// it makes port input register(PINn) internally pulled-up state that port output register(PORTn) outputs 1(high)
	PORTA = 0xFF;
	// it makes port input register(PINn) high-impedence state that port output register(PORTn) outputs 0(low)
	// so that we can share the pin with other devices
	//PORTA = 0x00;

	// uses all pins on PortD for output
	DDRD = 0xFF;
*/
	DDRA = 0xFF;  // uses all pins on PortA for output
	DDRD = 0xFF;  // uses all pins on PortD for output
}
