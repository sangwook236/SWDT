#include "spi.h"
#include <avr/interrupt.h>
#include <avr/sfr_defs.h>


#if defined(__SWL_AVR__USE_SPI_INTERRUPT)
volatile uint8_t spi_is_transmit_complete;

ISR(SPI_STC_vect)
{
	spi_is_transmit_complete = 1;
}

uint8_t spi_is_transmit_busy()
{
    return SPCR & _BV(SPIE);
}
#endif  // __SWL_AVR__USE_SPI_INTERRUPT

uint8_t spi_is_master()
{
	//return (SPCR & _BV(MSTR)) == _BV(MSTR);
	return bit_is_set(SPCR, MSTR);
}

uint8_t spi_master_transmit_a_byte(const uint8_t byte)
{
	SPDR = byte;
#if defined(__SWL_AVR__USE_SPI_INTERRUPT)
	while (!spi_is_transmit_complete) ;
	spi_is_transmit_complete = 0;
#else
	//while ((SPSR & _BV(SPIF)) != _BV(SPIF)) ;
	loop_until_bit_is_set(SPSR, SPIF);
#endif  // __SWL_AVR__USE_SPI_INTERRUPT

	const uint8_t val = SPDR;  // A byte received.
	return val;
}

void spi_master_transmit_bytes(const uint8_t *buf, const uint16_t lengthToBeWritten)
{
	uint8_t *ptr = (uint8_t *)buf;
	uint16_t len = lengthToBeWritten;

	while (len)
	{
		spi_master_transmit_a_byte((uint8_t)(*ptr++));
		--len;
	}
}

void spi_master_transmit_a_string(const uint8_t *str)
{
	while (*str)
	{
		spi_master_transmit_a_byte((uint8_t)(*str));
		++str;
	}
}

uint8_t spi_master_receive_a_byte()
{
	return spi_master_transmit_a_byte(0xFF);  // Dummy.
	//return spi_master_transmit_a_byte(0x00);  // Dummy.
}

void spi_master_receive_bytes(uint8_t *buf, const uint16_t lengthToBeRead)
{
	uint8_t *ptr = (uint8_t *)buf;
	uint16_t len = lengthToBeRead;

	while (len)
	{
		*ptr++ = spi_master_transmit_a_byte(0xFF);
		//*ptr++ = spi_master_transmit_a_byte(0x00);
		--len;
	}
}
