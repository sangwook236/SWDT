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

void spi_init_as_master()
{
	//SPSR = 0x00;
	//SPDR = 0;
	//SPCR = 0x00;  // must not set SPCR

	// check whether PB0 is an input
	if (bit_is_clear(DDRB, PB0))
	{
	    // if yes, activate the pull-up
	    PORTB |= _BV(PB0);
	}

	// switch SCK and MOSI pins to output mode
	DDRB |= _BV(PB1) | _BV(PB2);
	//PORTB &= ~(_BV(PB2));  // enable MOSI high-impedence

	// switch MISO pin to input mode
	DDRB &= ~(_BV(PB3));
	//PORTB |= _BV(PB3);  // enable MISO pull-up

	//PORTB &= 0xF0;  // make PB0 ~ PB3 high-impedence
	//PORTB |= 0x0F;  // make PB0 ~ PB3 pull-up

	// activate the SPI hardware
	//	SPIE: SPI interrupt enable
	//	SPE: SPI enable
#if defined(__SWL_AVR__USE_SPI_INTERRUPT)
	SPCR = _BV(SPIE) | _BV(SPE) | _BV(MSTR);
#else
	SPCR = _BV(SPE) | _BV(MSTR);
#endif  // __SWL_AVR__USE_SPI_INTERRUPT

	// clear status flags: the SPIF & WCOL flags
	uint8_t tmp;
	tmp = SPSR;
	tmp = SPDR;
}

void spi_init_as_slave()
{
	//SPSR = 0x00;
	//SPDR = 0;

	//SPCR = 0x00;  // must not set SPCR

	//PORTB = 0x00;

	// switch MOSI pin to input mode (?)
	DDRB &= ~(_BV(PB2));
	// enable MOSI pull-up
	PORTB |= _BV(PB2);

	// switch MISO pin to output mode
	DDRB |= _BV(PB3);

	// activate the SPI hardware
	//	SPIE: SPI interrupt enable
	//	SPE: SPI enable
#if defined(__SWL_AVR__USE_SPI_INTERRUPT)
	SPCR = _BV(SPIE) | _BV(SPE);
#else
	SPCR = _BV(SPE);
#endif  // __SWL_AVR__USE_SPI_INTERRUPT

	// clear status flags: the SPIF & WCOL flags
	uint8_t tmp;
	tmp = SPSR;
	tmp = SPDR;
}

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
	return SPDR;  // a byte received
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
	return spi_master_transmit_a_byte(0xFF);  // dummy
	//return spi_master_transmit_a_byte(0x00);  // dummy
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
