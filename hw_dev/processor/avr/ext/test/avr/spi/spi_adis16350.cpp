#include "spi_adis16350.h"
#include "spi.h"
#include <avr/io.h>
#include <util/delay.h>


//-----------------------------------------------------------------------------
// for ADIS16350

void adis16350_chip_select()
{
	PORTB &= ~(_BV(PB0));
}

void adis16350_chip_deselect()
{
	PORTB |= _BV(PB0);
}

void adis16350_init(const uint8_t is_master)
{
	// Step 1.
	//PORTB = 0x00;
	if (is_master)
	{
		// Switch SS pin to output mode.
		DDRB |= _BV(PB0);
		// Check whether PB0 is an input.
		if (bit_is_clear(DDRB, PB0))
		{
		    // If yes, activate the pull-up.
		    PORTB |= _BV(PB0);
		}

		// Switch SCK and MOSI pins to output mode.
		DDRB |= _BV(PB1) | _BV(PB2);
		//PORTB &= ~(_BV(PB2));  // Enable MOSI high-impedence.

		// Switch MISO pin to input mode.
		DDRB &= ~(_BV(PB3));
		//PORTB |= _BV(PB3);  // Enable MISO pull-up.

		//PORTB &= 0xF0;  // Make PB0 ~ PB3 high-impedence.
		//PORTB |= 0x0F;  // Make PB0 ~ PB3 pull-up.
	}
	else
	{
		// Switch MOSI pin to input mode. (?)
		DDRB &= ~(_BV(PB2));
		// Enable MOSI pull-up.
		PORTB |= _BV(PB2);

		// Switch MISO pin to output mode.
		DDRB |= _BV(PB3);
	}

	// Step 2: Initialize ADIS16350 SPI module.
	adis16350_chip_deselect();

	// Step 3.
	//SPSR = 0x00;
	//SPDR = 0;
	//SPCR = 0x00;  // Must not set SPCR.

	// Data order: if 0, MSB to LSB. if 1, LSB to MSB.
	SPCR &= ~(_BV(DORD));  // The MSB is the first bit transmitted and received.

	// Clock polarity: if 0, leading edge = rising edge. if 1, leading edge = falling edge.
	// Clock phase: if 0, sample at leading edge. if 1, sample at trailing edge.

	// SPI mode		CPOL	CPHA	at leading edge		at trailing edge
	//--------------------------------------------------------------------
	// 0			0		0		sample (rising)		setup (falling)
	// 1			0		1		setup (rising)		sample (falling)
	// 2			1		0		sample (falling)	setup (rising)
	// 3			1		1		setup (falling)		sample (rising)
	SPCR |= _BV(CPOL);
	SPCR |= _BV(CPHA);

	// SCK clock rate
	// SPI2X	SPR1	SPR0	SCK frequency
	//-----------------------------------------
	// 0		0		0		Fosc / 4
	// 0		0		1		Fosc / 16
	// 0		1		0		Fosc / 64
	// 0		1		1		Fosc / 128
	// 1		0		0		Fosc / 2
	// 1		0		1		Fosc / 8
	// 1		1		0		Fosc / 32
	// 1		1		1		Fosc / 64

	// Sampling period of ADIS16350.
	//	SPI SCK <= 2.5 MHz for FAST MODE (SMPL_PRD register <= 0x09).
	SPSR |= _BV(SPI2X);
	SPCR &= ~(_BV(SPR1));
	SPCR |= _BV(SPR0);
	//	SPI SCK <= 1 MHz for NORMAL MODE (SMPL_PRD register > 0x09).
	//SPSR &= ~(_BV(SPI2X));
	//SPCR &= ~(_BV(SPR1));
	//SPCR |= _BV(SPR0);

	// Step 4.
	// Activate the SPI hardware.
	//	SPIE: SPI interrupt enable.
	//	SPE: SPI enable.
#if defined(__SWL_AVR__USE_SPI_INTERRUPT)
	if (is_master)
		SPCR |= _BV(SPIE) | _BV(SPE) | _BV(MSTR);
	else
		SPCR |= _BV(SPIE) | _BV(SPE);
#else
	if (is_master)
		SPCR |= _BV(SPE) | _BV(MSTR);
	else
		SPCR |= _BV(SPE);
#endif  // __SWL_AVR__USE_SPI_INTERRUPT

	// Clear status flags: the SPIF & WCOL flags.
	uint8_t dummy;
	dummy = SPSR;
	dummy = SPDR;
}

int adis16350_write_a_register(const uint8_t addr, const uint16_t word)
{
	const uint8_t OP_CODE = 0x80;  // Write.

	spi_disable_interrupt();

	adis16350_chip_select();
	spi_master_transmit_a_byte(OP_CODE | ((addr + 1) & 0x3F));
	spi_master_transmit_a_byte((word >> 8) & 0xFF);
	adis16350_chip_deselect();

	_delay_ms(10);

	adis16350_chip_select();
	spi_master_transmit_a_byte(OP_CODE | (addr & 0x3F));
	spi_master_transmit_a_byte(word & 0xFF);
	adis16350_chip_deselect();

	spi_enable_interrupt();

	return 1;
}

int adis16350_read_a_register(const uint8_t addr, uint16_t *word)
{
	const uint8_t OP_CODE = 0x00;  // Read.

	spi_disable_interrupt();

	adis16350_chip_select();
	const uint8_t upper = spi_master_transmit_a_byte(OP_CODE | (addr & 0x3F));
	const uint8_t lower = spi_master_transmit_a_byte(0x00);  // Dummy.
	//const uint8_t upper = spi_master_transmit_a_byte(0x00);  // Dummy.
	//const uint8_t lower = spi_master_transmit_a_byte(0x00);  // Dummy.
	adis16350_chip_deselect();

	spi_enable_interrupt();

	*word = ((upper << 8) & 0xFF00) | (lower & 0x00FF);

	return 1;
}
