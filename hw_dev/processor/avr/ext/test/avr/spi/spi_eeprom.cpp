#include "spi_eeprom.h"
#include "spi.h"
#include <avr/io.h>


//-----------------------------------------------------------------------------
// For 93C46/93C56/93C66: use 8-bit access.
// REF [book] >> "AVR ATmega128 ������", ������ ��, Ohm��. pp. 636~644.

static const uint8_t SPI_START_BIT_93CXX = 0x01 << 3;

void ee93Cxx_chip_select()
{
	PORTB |= _BV(PB0);
}

void ee93Cxx_chip_deselect()
{
	PORTB &= ~(_BV(PB0));
}

void ee93Cxx_init(const uint8_t is_master)
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

	// Step 2: Initialize EE93Cxx SPI module.
	ee93Cxx_chip_deselect();

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
	SPSR &= ~(_BV(SPI2X));
	SPCR &= ~(_BV(SPR1));
	SPCR |= _BV(SPR0);

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


int ee93Cxx_set_write_enable()
{
	const uint8_t OP_CODE = 0x00;  // Write enable.
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
	const uint8_t OP_CODE = 0x00;  // Write disable.
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
	const uint8_t OP_CODE = 0x01;  // Write.

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
	const uint8_t OP_CODE = 0x02;  // Read.

	spi_disable_interrupt();
	ee93Cxx_chip_select();
	spi_master_transmit_a_byte(SPI_START_BIT_93CXX | ((OP_CODE << 1) & 0x06) | ((eeaddr >> 8) & 0x01));
	spi_master_transmit_a_byte(eeaddr & 0x00FF);
	const uint8_t upper = spi_master_transmit_a_byte(0x00);  // Dummy.
	const uint8_t lower = spi_master_transmit_a_byte(0x00);  // Dummy.
	ee93Cxx_chip_deselect();
	spi_disable_interrupt();

	*byte = (upper << 1) + (lower >> 7);

	return 1;
}

int ee93Cxx_erase(const uint16_t eeaddr)
{
	const uint8_t OP_CODE = 0x03;  // Erase.

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
	const uint8_t OP_CODE = 0x00;  // Erase all.
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
	const uint8_t OP_CODE = 0x00;  // Write all.
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

void ee25xxx_chip_select()
{
	PORTB &= ~(_BV(PB0));
}

void ee25xxx_chip_deselect()
{
	PORTB |= _BV(PB0);
}

void ee25xxx_init(const uint8_t is_master)
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

	// Step 2: Iinitialize EE25xxx SPI module.
	ee25xxx_chip_deselect();

	// Step 3.
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
	SPCR &= ~(_BV(CPOL));
	SPCR &= ~(_BV(CPHA));
	//SPCR |= _BV(CPOL);
	//SPCR |= _BV(CPHA);

	// SCK clock rate.
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
	spi_master_transmit_a_byte(EE25xxx_INSTRUCTION_WREN & 0x07);  // Write enable.
	ee25xxx_chip_deselect();
	spi_enable_interrupt();

	return 1;
}

int ee25xxx_set_write_disable()
{
	spi_disable_interrupt();
	ee25xxx_chip_select();
	spi_master_transmit_a_byte(EE25xxx_INSTRUCTION_WRDI & 0x07);  // Write disable.
	ee25xxx_chip_deselect();
	spi_enable_interrupt();

	return 1;
}

int ee25xxx_write_status_register(const uint8_t byte)
{
	spi_disable_interrupt();
	ee25xxx_chip_select();
	spi_master_transmit_a_byte(EE25xxx_INSTRUCTION_WRSR & 0x07);  // Write status register.
	spi_master_transmit_a_byte(byte);
	ee25xxx_chip_deselect();
	spi_enable_interrupt();

	return 1;
}

int ee25xxx_read_status_register(uint8_t *byte)
{
	spi_disable_interrupt();
	ee25xxx_chip_select();
	spi_master_transmit_a_byte(EE25xxx_INSTRUCTION_RDSR & 0x07);  // Read status register.
	*byte = spi_master_transmit_a_byte(0xFF);  // Dummy.
	ee25xxx_chip_deselect();
	spi_enable_interrupt();

	return 1;
}

int ee25xxx_write_a_byte(const uint16_t eeaddr, const uint8_t byte)
{
	spi_disable_interrupt();
	ee25xxx_chip_select();
	spi_master_transmit_a_byte(EE25xxx_INSTRUCTION_WRITE & 0x07);  // Write data.
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
	spi_master_transmit_a_byte(EE25xxx_INSTRUCTION_READ & 0x07);  // Read data.
	spi_master_transmit_a_byte((eeaddr >> 8) & 0x00FF);
	spi_master_transmit_a_byte(eeaddr & 0x00FF);
	*byte = spi_master_transmit_a_byte(0xFF);  // Dummy.
	ee25xxx_chip_deselect();
	spi_enable_interrupt();

	return 1;
}
