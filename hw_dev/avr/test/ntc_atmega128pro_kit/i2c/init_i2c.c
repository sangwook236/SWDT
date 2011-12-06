#include <util/twi.h>
#include <avr/interrupt.h>
#include <avr/sfr_defs.h>
#include <math.h>


ISR(TWI_vect)
{
	TWCR |= _BV(TWINT);  // TWI interrupt flag clear
}

void initI2c()
{
	TWDR = 0;

	TWCR = 0x00;
	TWCR |= _BV(TWEA);  // TWI enable acknowledge bit
	//TWCR |= _BV(TWSTA);  // TWI start condition bit
	//TWCR |= _BV(TWSTO);  // TWI stop condition bit
	TWCR |= _BV(TWEN);  // TWI enable bit
	TWCR |= _BV(TWIE);  // TWI interrupt enable

	TWSR = 0x00;
	// TWI prescaler
	const uint8_t twi_prescaler = 0;
	switch (twi_prescaler) {
	case 0:
		TWSR &= ~(_BV(TWPS1));
		TWSR &= ~(_BV(TWPS0));
		break;
	case 1:
		TWSR &= ~(_BV(TWPS1));
		TWSR |= _BV(TWPS0);
		break;
	case 2:
		TWSR |= _BV(TWPS1);
		TWSR &= ~(_BV(TWPS0));
		break;
	case 3:
		TWSR |= _BV(TWPS1);
		TWSR |= _BV(TWPS0);
		break;
	}

	// serial clock rate
	//	Fscl = Fosc / (16 + 2 * TWBR * 4^TWPS)
	//	TWBR = (Fosc / Fscl - 16) / (2 * 4^TWPS)
	const uint32_t Fscl = 400000UL;  // fast mode: 400 kbps
	TWBR = (uint8_t)((float)(F_CPU / Fscl - 16) / (float)(2 * pow(4, twi_prescaler)));

	TWAR = 0x02;  // TWI slave address bits: 0b0000 001?
	TWAR |= _BV(TWGCE);  // TWI general call recognition enable bit
}

