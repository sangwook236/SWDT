#include <avr/interrupt.h>
#include <avr/sfr_defs.h>
#include <util/delay.h>

/*
// If an unexpected interrupt occurs (interrupt is enabled and no handler is installed, which usually indicates a bug),
// then the default action is to reset the device by jumping to the reset vector.
// You can override this by supplying a function named __vector_default which should be defined with ISR()
ISR(__vector_default)
{
    // user code here
}
*/

ISR(INT0_vect)
{
	PORTA = 0x01;
	_delay_ms(500);  // 0.5 sec delay
	PORTA = 0x10;
	_delay_ms(500);  // 0.5 sec delay
	PORTA = 0x00;
	_delay_ms(500);  // 0.5 sec delay
}

ISR(INT1_vect)
{
	PORTA = 0x02;
	_delay_ms(500);  // 0.5 sec delay
	PORTA = 0x20;
	_delay_ms(500);  // 0.5 sec delay
	PORTA = 0x00;
	_delay_ms(500);  // 0.5 sec delay
}

ISR(INT2_vect)
{
	PORTA = 0x04;
	_delay_ms(500);  // 0.5 sec delay
	PORTA = 0x40;
	_delay_ms(500);  // 0.5 sec delay
	PORTA = 0x00;
	_delay_ms(500);  // 0.5 sec delay
}

ISR(INT3_vect)
{
	PORTA = 0x08;
	_delay_ms(500);  // 0.5 sec delay
	PORTA = 0x80;
	_delay_ms(500);  // 0.5 sec delay
	PORTA = 0x00;
	_delay_ms(500);  // 0.5 sec delay
}

/*
ISR(INT4_vect)
{
}

ISR(INT5_vect)
{
}

ISR(INT6_vect)
{
}

ISR(INT7_vect)
{
}
*/

void initExtInt()
{
	EICRA = 0x00;
	// ISCn0	ISCn1	(when n = 0 ~ 3)
	// 0		0		INTn is triggered by low-level input
	// 0		1		reserved
	// 1		0		INTn is asynchronously triggered by falling-edge input
	// 1		1		INTn is asynchronously triggered by rising-edge input
	EICRA |= _BV(ISC00);
	EICRA &= ~(_BV(ISC01));
	EICRA |= _BV(ISC10);
	EICRA &= ~(_BV(ISC11));
	EICRA |= _BV(ISC20);
	EICRA &= ~(_BV(ISC21));
	EICRA |= _BV(ISC30);
	EICRA &= ~(_BV(ISC31));

	EICRB = 0x00;
	// ISCn0	ISCn1	(when n = 4 ~ 7)
	// 0		0		INTn is triggered by low-level input
	// 0		1		INTn is triggered by falling-edge or rising-edge input
	// 1		0		INTn is triggered by falling-edge input
	// 1		1		INTn is triggered by rising-edge input
	//EICRB &= ~(_BV(ISC40));
	//EICRB |= _BV(ISC41);
	//EICRB &= ~(_BV(ISC50));
	//EICRB |= _BV(ISC51);
	//EICRB &= ~(_BV(ISC60));
	//EICRB |= _BV(ISC61);
	//EICRB &= ~(_BV(ISC70));
	//EICRB |= _BV(ISC71);

	EIMSK = 0x00;
	EIMSK |= _BV(INT0);  // enables EINT0 interrupt
	EIMSK |= _BV(INT1);  // enables EINT1 interrupt
	EIMSK |= _BV(INT2);  // enables EINT2 interrupt
	EIMSK |= _BV(INT3);  // enables EINT3 interrupt
	//EIMSK |= _BV(INT4);  // enables EINT4 interrupt
	//EIMSK |= _BV(INT5);  // enables EINT5 interrupt
	//EIMSK |= _BV(INT6);  // enables EINT6 interrupt
	//EIMSK |= _BV(INT7);  // enables EINT7 interrupt

}
