#include "spi_adis16350.h"
#include "spi.h"
#include <avr/io.h>
#include <util/delay.h>


//-----------------------------------------------------------------------------
// for ADIS16350

void adis16350_init()
{
	//SPCR = 0x00;

	// data order: if 0, MSB to LSB. if 1, LSB to MSB.
	SPCR &= ~(_BV(DORD));  // the MSB is the first bit transmitted and received

	// SPI mode		CPOL	CPHA	at leading edge		at trailing edge
	//--------------------------------------------------------------------
	// 0			0		0		sample (rising)		setup (falling)
	// 1			0		1		setup (rising)		sample (falling)
	// 2			1		0		sample (falling)	setup (rising)
	// 3			1		1		setup (falling)		sample (rising)
	SPCR |= _BV(CPOL);  // clock polarity: if 0, leading edge = rising edge. if 1, leading edge = falling edge.
	SPCR |= _BV(CPHA);  // clock phase: if 0, sample at leading edge. if 1, sample at trailing edge

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

	// sampling period of ADIS16350
	//	SPI SCK <= 2.5 MHz for FAST MODE (SMPL_PRD register <= 0x09)
	//SPSR |= _BV(SPI2X);
	//SPCR &= ~(_BV(SPR1));
	//SPCR |= _BV(SPR0);
	//	SPI SCK <= 1 MHz for NORMAL MODE (SMPL_PRD register > 0x09)
	SPSR &= ~(_BV(SPI2X));
	SPCR |= _BV(SPR1);
	SPCR |= _BV(SPR0);

	// switch SS pin to output mode
	DDRB |= _BV(PB0);

	//
	spi_init_as_master();
}

void adis16350_chip_select()
{
	PORTB &= ~(_BV(PB0));
}

void adis16350_chip_deselect()
{
	PORTB |= _BV(PB0);
}

int adis16350_write_a_register(const uint8_t addr, const uint16_t word)
{
	const uint8_t OP_CODE = 0x80;  // write

	spi_disable_interrupt();
	adis16350_chip_select();
	// FIXME [check] >>
	spi_master_transmit_a_byte(OP_CODE | (addr & 0x3F));
	spi_master_transmit_a_byte(word);
	spi_master_transmit_a_byte((word >> 8) & 0x0F);
/*
	spi_master_transmit_a_byte(OP_CODE | (addr & 0x3F));
	spi_master_transmit_bytes((uint8_t *)&word, 2);
*/
/*
	spi_master_transmit_a_byte(OP_CODE | (addr & 0x3F));
	spi_master_transmit_a_byte(word);
	spi_master_transmit_a_byte(OP_CODE | ((addr + 1) & 0x3F));
	spi_master_transmit_a_byte((word >> 8) & 0x0F);
*/
	adis16350_chip_deselect();
	spi_disable_interrupt();

	return 1;
}

int adis16350_read_a_register(const uint8_t addr, uint16_t *word)
{
	const uint8_t OP_CODE = 0x00;  // read

	spi_disable_interrupt();
	adis16350_chip_select();
	// FIXME [check] >>
	spi_master_transmit_a_byte(OP_CODE | (addr & 0x3F));
	spi_master_transmit_a_byte(0x00);  // dummy
	const uint8_t upper = spi_master_transmit_a_byte(0x00);  // dummy
	const uint8_t lower = spi_master_transmit_a_byte(0x00);  // dummy
/*
	spi_master_transmit_a_byte(OP_CODE | (addr & 0x3F));
	spi_master_transmit_a_byte(0x00);
	spi_master_receive_bytes((uint8_t *)word, 2);
*/
	adis16350_chip_deselect();
	spi_disable_interrupt();

	*word = ((upper << 8) & 0xFF00) | (lower & 0x00FF);
	PORTA = upper;
	_delay_ms(500);
	PORTA = lower;
	_delay_ms(500);

	return 1;
}
