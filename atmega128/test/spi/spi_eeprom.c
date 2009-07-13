#include "spi_eeprom.h"
#include "spi.h"
#include <avr/io.h>


//-----------------------------------------------------------------------------
// for 93C46/93C56/93C66: use 8-bit access
// [ref] "AVR ATmega128 ¸¶½ºÅÍ", À±´ö¿ë Àú, Ohm»ç. pp. 636~644

static const uint8_t SPI_START_BIT_93CXX = 0x01 << 3;

void ee93Cxx_init()
{
	//SPCR = 0x00;

	// data order: if 0, MSB to LSB. if 1, LSB to MSB.
	SPCR &= ~(_BV(DORD));  // the MSB is the first bit transmitted and received

	// clock polarity: if 0, leading edge = rising edge. if 1, leading edge = falling edge.
	// clock phase: if 0, sample at leading edge. if 1, sample at trailing edge

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
	SPSR &= ~(_BV(SPI2X));
	SPCR &= ~(_BV(SPR1));
	SPCR |= _BV(SPR0);

	// switch SS pin to output mode
	DDRB |= _BV(PB0);

	//
	spi_init_as_master();
}

void ee93Cxx_chip_select()
{
	PORTB |= _BV(PB0);
}

void ee93Cxx_chip_deselect()
{
	PORTB &= ~(_BV(PB0));
}

int ee93Cxx_set_write_enable()
{
	const uint8_t OP_CODE = 0x00;  // write enable
	const uint16_t addr = 0x0180;

	spi_disable_interrupt();
	ee93Cxx_chip_select();
	spi_master_transmit_a_byte(SPI_START_BIT_93CXX | ((OP_CODE << 1) & 0x06) | ((addr >> 8) & 0x01));
	spi_master_transmit_a_byte(addr & 0x00FF);
	ee93Cxx_chip_deselect();
	spi_disable_interrupt();

	return 1;
}

int ee93Cxx_set_write_disable()
{
	const uint8_t OP_CODE = 0x00;  // write disable
	const uint16_t addr = 0x0000;

	spi_disable_interrupt();
	ee93Cxx_chip_select();
	spi_master_transmit_a_byte(SPI_START_BIT_93CXX | ((OP_CODE << 1) & 0x06) | ((addr >> 8) & 0x01));
	spi_master_transmit_a_byte(addr & 0x00FF);
	ee93Cxx_chip_deselect();
	spi_disable_interrupt();

	return 1;
}

int ee93Cxx_write_a_byte(const uint16_t eeaddr, const uint8_t byte)
{
	const uint8_t OP_CODE = 0x01;  // write

	spi_disable_interrupt();
	ee93Cxx_chip_select();
	spi_master_transmit_a_byte(SPI_START_BIT_93CXX | ((OP_CODE << 1) & 0x06) | ((eeaddr >> 8) & 0x01));
	spi_master_transmit_a_byte(eeaddr & 0x00FF);
	spi_master_transmit_a_byte(byte);
	ee93Cxx_chip_deselect();
	spi_disable_interrupt();

	return 1;
}

int ee93Cxx_read_a_byte(const uint16_t eeaddr, uint8_t *byte)
{
	const uint8_t OP_CODE = 0x02;  // read

	spi_disable_interrupt();
	ee93Cxx_chip_select();
	spi_master_transmit_a_byte(SPI_START_BIT_93CXX | ((OP_CODE << 1) & 0x06) | ((eeaddr >> 8) & 0x01));
	spi_master_transmit_a_byte(eeaddr & 0x00FF);
	const uint8_t upper = spi_master_transmit_a_byte(0x00);  // dummy
	const uint8_t lower = spi_master_transmit_a_byte(0x00);  // dummy
	ee93Cxx_chip_deselect();
	spi_disable_interrupt();

	*byte = (upper << 1) + (lower >> 7);

	return 1;
}

int ee93Cxx_erase(const uint16_t eeaddr)
{
	const uint8_t OP_CODE = 0x03;  // erase

	spi_disable_interrupt();
	ee93Cxx_chip_select();
	spi_master_transmit_a_byte(SPI_START_BIT_93CXX | ((OP_CODE << 1) & 0x06) | ((eeaddr >> 8) & 0x01));
	spi_master_transmit_a_byte(eeaddr & 0x00FF);
	ee93Cxx_chip_deselect();
	spi_disable_interrupt();

	return 1;
}

int ee93Cxx_erase_all()
{
	const uint8_t OP_CODE = 0x00;  // erase all
	const uint16_t addr = 0x0100;

	spi_disable_interrupt();
	ee93Cxx_chip_select();
	spi_master_transmit_a_byte(SPI_START_BIT_93CXX | ((OP_CODE << 1) & 0x06) | ((addr >> 8) & 0x01));
	spi_master_transmit_a_byte(addr & 0x00FF);
	ee93Cxx_chip_deselect();
	spi_disable_interrupt();

	return 1;
}

int ee93Cxx_write_all(const uint8_t byte)
{
	const uint8_t OP_CODE = 0x00;  // write all
	const uint16_t addr = 0x0080;

	spi_disable_interrupt();
	ee93Cxx_chip_select();
	spi_master_transmit_a_byte(SPI_START_BIT_93CXX | ((OP_CODE << 1) & 0x06) | ((addr >> 8) & 0x01));
	spi_master_transmit_a_byte(addr & 0x00FF);
	spi_master_transmit_a_byte(byte);
	ee93Cxx_chip_deselect();
	spi_disable_interrupt();

	return 1;
}

//-----------------------------------------------------------------------------
// for 25128/25256

#define EE25xxx_INSTRUCTION_WREN 0x06
#define EE25xxx_INSTRUCTION_WRDI 0x04
#define EE25xxx_INSTRUCTION_RDSR 0x05
#define EE25xxx_INSTRUCTION_WRSR 0x01
#define EE25xxx_INSTRUCTION_READ 0x03
#define EE25xxx_INSTRUCTION_WRITE 0x02

void ee25xxx_init()
{
	//SPCR = 0x00;

	// data order: if 0, MSB to LSB. if 1, LSB to MSB.
	SPCR &= ~(_BV(DORD));  // the MSB is the first bit transmitted and received

	// clock polarity: if 0, leading edge = rising edge. if 1, leading edge = falling edge.
	// clock phase: if 0, sample at leading edge. if 1, sample at trailing edge

	// SPI mode		CPOL	CPHA	at leading edge		at trailing edge
	//--------------------------------------------------------------------
	// 0			0		0		sample (rising)		setup (falling)
	// 1			0		1		setup (rising)		sample (falling)
	// 2			1		0		sample (falling)	setup (rising)
	// 3			1		1		setup (falling)		sample (rising)
	SPCR &= ~(_BV(CPOL));
	SPCR &= ~(_BV(CPHA));
	//SPCR |= _BV(CPOL);
	//SPCR |= _BV(CPHA);

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
	SPSR &= ~(_BV(SPI2X));
	SPCR &= ~(_BV(SPR1));
	SPCR &= ~(_BV(SPR0));

	// switch SS pin to output mode
	DDRB |= _BV(PB0);

	//
	spi_init_as_master();
}

void ee25xxx_chip_select()
{
	PORTB &= ~(_BV(PB0));
}

void ee25xxx_chip_deselect()
{
	PORTB |= _BV(PB0);
}

int ee25xxx_is_ready()
{
	uint8_t status = 0xFF;
	ee25xxx_read_status_register(&status);
	return (status & 0x01) == 0x00;
}

int ee25xxx_set_write_enable()
{
	spi_disable_interrupt();
	ee25xxx_chip_select();
	spi_master_transmit_a_byte(EE25xxx_INSTRUCTION_WREN & 0x07);  // write enable
	ee25xxx_chip_deselect();
	spi_enable_interrupt();

	return 1;
}

int ee25xxx_set_write_disable()
{
	spi_disable_interrupt();
	ee25xxx_chip_select();
	spi_master_transmit_a_byte(EE25xxx_INSTRUCTION_WRDI & 0x07);  // write disable
	ee25xxx_chip_deselect();
	spi_enable_interrupt();

	return 1;
}

int ee25xxx_write_status_register(const uint8_t byte)
{
	spi_disable_interrupt();
	ee25xxx_chip_select();
	spi_master_transmit_a_byte(EE25xxx_INSTRUCTION_WRSR & 0x07);  // write status register
	spi_master_transmit_a_byte(byte);
	ee25xxx_chip_deselect();
	spi_enable_interrupt();

	return 1;
}

int ee25xxx_read_status_register(uint8_t *byte)
{
	spi_disable_interrupt();
	ee25xxx_chip_select();
	spi_master_transmit_a_byte(EE25xxx_INSTRUCTION_RDSR & 0x07);  // read status register
	*byte = spi_master_transmit_a_byte(0xFF);  // dummy
	ee25xxx_chip_deselect();
	spi_enable_interrupt();

	return 1;
}

int ee25xxx_write_a_byte(const uint16_t eeaddr, const uint8_t byte)
{
	spi_disable_interrupt();
	ee25xxx_chip_select();
	spi_master_transmit_a_byte(EE25xxx_INSTRUCTION_WRITE & 0x07);  // write data
	spi_master_transmit_a_byte((eeaddr >> 8) & 0x00FF);
	spi_master_transmit_a_byte(eeaddr & 0x00FF);
	spi_master_transmit_a_byte(byte);
	ee25xxx_chip_deselect();
	spi_enable_interrupt();

	return 1;
}

int ee25xxx_read_a_byte(const uint16_t eeaddr, uint8_t *byte)
{
	spi_disable_interrupt();
	ee25xxx_chip_select();
	spi_master_transmit_a_byte(EE25xxx_INSTRUCTION_READ & 0x07);  // read data
	spi_master_transmit_a_byte((eeaddr >> 8) & 0x00FF);
	spi_master_transmit_a_byte(eeaddr & 0x00FF);
	*byte = spi_master_transmit_a_byte(0xFF);  // dummy
	ee25xxx_chip_deselect();
	spi_enable_interrupt();

	return 1;
}
