#include "i2c.h"
#include <avr/interrupt.h>
#include <avr/sfr_defs.h>
#include <util/twi.h>
#include <math.h>


ISR(TWI_vect)
{
	TWCR |= _BV(TWINT);  // TWI interrupt flag clear
}

void initI2c(const uint8_t is_fast_mode)
{
	TWDR = 0;
	TWCR = 0x00;

	TWSR = 0x00;
	// TWI prescaler
	const uint8_t prescaler = 1;
	switch (prescaler)
	{
	case 4:
		TWSR &= ~(_BV(TWPS1));
		TWSR |= _BV(TWPS0);
		break;
	case 16:
		TWSR |= _BV(TWPS1);
		TWSR &= ~(_BV(TWPS0));
		break;
	case 64:
		TWSR |= _BV(TWPS1);
		TWSR |= _BV(TWPS0);
		break;
	case 1:
	default:
		TWSR &= ~(_BV(TWPS1));
		TWSR &= ~(_BV(TWPS0));
		break;
	}

	// serial clock rate
	//	Fscl = Fosc / (16 + 2 * TWBR * 4^TWPS)
	//	TWBR = (Fosc / Fscl - 16) / (2 * 4^TWPS)
#if F_CPU < 3600000UL
 	TWBR = 10;  /// smallest TWBR value
#else
	// regular mode: 100 kbps, fast mode: 400 kbps
	const uint32_t Fscl = is_fast_mode ? 400000UL : 100000UL;
	TWBR = (uint8_t)((float)(F_CPU / Fscl - 16) / (float)(2 * prescaler));
#endif

	TWAR = 0x00;  // TWI slave address bits: 0b0000 000?
	TWAR |= _BV(TWGCE);  // TWI general call recognition enable bit
}

//-----------------------------------------------------------------------------
//

I2C_STATUS i2c_start()
{
	// send start condition
	TWCR = _BV(TWINT) | _BV(TWSTA) | _BV(TWEN);
	// wait for transmission
	while ((TWCR & _BV(TWINT)) == 0) ;

	const uint8_t twsr = (TWSR & TW_STATUS_MASK);
	switch (twsr)
	{
	case TW_REP_START:  // OK, but should not happen
	case TW_START:
		return I2C_OK;  // OK
	case TW_MT_ARB_LOST:
		return I2C_ERR_ARB_LOST;  // begin
	default:
		return I2C_ERR_QUIT_WITHOUT_STOP;  // error: not in start condition
		// NB: do not send stop condition
	}
}

I2C_STATUS i2c_repeated_start()
{
	// send (repeated) start condition
	TWCR = _BV(TWINT) | _BV(TWSTA) | _BV(TWEN);
	// wait for transmission
	while ((TWCR & _BV(TWINT)) == 0) ;

	const uint8_t twsr = (TWSR & TW_STATUS_MASK);
	switch (twsr)
	{
	case TW_START:  // OK, but should not happen
	case TW_REP_START:
		return I2C_OK;  // OK
	case TW_MT_ARB_LOST:
		return I2C_ERR_ARB_LOST;  // begin
	default:
		return I2C_ERR_QUIT_WITH_STOP;  // must send stop condition
	}
}

void i2c_stop()
{
	// send stop condition
	TWCR = _BV(TWINT) | _BV(TWSTO) | _BV(TWEN);
}

I2C_STATUS i2c_sla_w(const uint8_t sla)
{
	// send SLA+W
	TWDR = (sla & 0xFE) | TW_WRITE;
	// clear interrupt to start transmission
	TWCR = _BV(TWINT) | _BV(TWEN);
	// wait for transmission
	while ((TWCR & _BV(TWINT)) == 0) ;

	const uint8_t twsr = (TWSR & TW_STATUS_MASK);
	switch (twsr)
	{
	case TW_MT_SLA_ACK:
		return I2C_OK;
	case TW_MT_SLA_NACK:  // nack during select: device busy writing
		return I2C_ERR_RESTART;  // restart
	case TW_MT_ARB_LOST:  // re-arbitrate
		return I2C_ERR_ARB_LOST;  // begin
	default:
		return I2C_ERR_QUIT_WITH_STOP;  // must send stop condition
	}
}

I2C_STATUS i2c_sla_r(const uint8_t sla)
{
	// send SLA+R
	TWDR = (sla & 0xFE) | TW_READ;
	// clear interrupt to start transmission
	TWCR = _BV(TWINT) | _BV(TWEN);
	// wait for transmission
	while ((TWCR & _BV(TWINT)) == 0) ;

	const uint8_t twsr = (TWSR & TW_STATUS_MASK);
	switch (twsr)
	{
	case TW_MR_SLA_ACK:
		return I2C_OK;
	case TW_MR_SLA_NACK:
		return I2C_QUIT;
	case TW_MR_ARB_LOST:
		return I2C_ERR_ARB_LOST;  // begin
	default:
		return I2C_ERR_QUIT_WITH_STOP;  // must send stop condition
	}
}

I2C_STATUS i2c_address(const uint8_t addr)
{
	// low 8 bits of addr
	TWDR = addr;
	// clear interrupt to start transmission
	TWCR = _BV(TWINT) | _BV(TWEN);
	// wait for transmission
	while ((TWCR & _BV(TWINT)) == 0) ;

	const uint8_t twsr = (TWSR & TW_STATUS_MASK);
	switch (twsr)
	{
	case TW_MT_DATA_ACK:
		return I2C_OK;
	case TW_MT_DATA_NACK:
		return I2C_QUIT;  // quit
	case TW_MT_ARB_LOST:
		return I2C_ERR_ARB_LOST;  // begin
	default:
		return I2C_ERR_QUIT_WITH_STOP;  // must send stop condition
	}
}

I2C_STATUS i2c_master_write_a_byte(const uint8_t byte)
{
	TWDR = byte;
	// clear interrupt to start transmission
	TWCR = _BV(TWINT) | _BV(TWEN);
	// wait for transmission
	while ((TWCR & _BV(TWINT)) == 0) ;

	const uint8_t twsr = (TWSR & TW_STATUS_MASK);
	switch (twsr)
	{
	case TW_MT_DATA_NACK:
		return I2C_ERR_QUIT_WITH_STOP;  // device write protected
	case TW_MT_DATA_ACK:
		return I2C_OK;
	default:
		return I2C_ERR_QUIT_WITH_STOP;  // must send stop condition
	}
}

I2C_STATUS i2c_master_write_bytes(const uint16_t bufLen, const uint8_t *buf, uint16_t *byteLenWritten)
{
	//for (uint16_t len = bufLen; len > 0; --len)
	for (uint16_t len = 0; len < bufLen; ++len)
	{
		TWDR = *buf++;
		// clear interrupt to start transmission
		TWCR = _BV(TWINT) | _BV(TWEN);
		// wait for transmission
		while ((TWCR & _BV(TWINT)) == 0) ;

		const uint8_t twsr = (TWSR & TW_STATUS_MASK);
		switch (twsr)
		{
		case TW_MT_DATA_NACK:
			return I2C_ERR_QUIT_WITH_STOP;  // device write protected
		case TW_MT_DATA_ACK:
			++(*byteLenWritten);
			break;
		default:
			return I2C_ERR_QUIT_WITH_STOP;  // must send stop condition
		}
	}

	return I2C_OK;
}

I2C_STATUS i2c_master_read_a_byte(uint8_t *byte)
{
	// clear interrupt to start reception
	TWCR = _BV(TWINT) | _BV(TWEN);  // send NACK
	// wait for reception
	while ((TWCR & _BV(TWINT)) == 0) ;

	const uint8_t twsr = (TWSR & TW_STATUS_MASK);
	switch (twsr)
	{
	case TW_MR_DATA_NACK:
	case TW_MR_DATA_ACK:
		*byte = TWDR;
		return I2C_OK;
	default:
		return I2C_ERR_QUIT_WITH_STOP;  // must send stop condition
	}
}

I2C_STATUS i2c_master_read_bytes(const uint16_t bufLen, uint8_t *buf, uint16_t *byteLenRead)
{
	uint8_t is_looping = 1;
	for (uint16_t len = bufLen; is_looping && len > 0; --len)
	{
		// clear interrupt to start reception
		if (len == 1)
			TWCR = _BV(TWINT) | _BV(TWEN);  // send NACK
		else
			TWCR = _BV(TWINT) | _BV(TWEN) | _BV(TWEA);  // send ACK
		// wait for reception
		while ((TWCR & _BV(TWINT)) == 0) ;

		const uint8_t twsr = (TWSR & TW_STATUS_MASK);
		switch (twsr)
		{
		case TW_MR_DATA_NACK:
			is_looping = 0;  // force end of loop
		case TW_MR_DATA_ACK:
			*buf++ = TWDR;
			++(*byteLenRead);
			break;
		default:
			return I2C_ERR_QUIT_WITH_STOP;  // must send stop condition
		}
	}

	return I2C_OK;
}
