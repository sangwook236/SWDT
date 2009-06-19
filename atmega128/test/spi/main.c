#include "spi_eeprom.h"
#include "spi_adis16350.h"
#include <avr/sleep.h>
#include <avr/interrupt.h>
#include <util/delay.h>


void initSystem();
void test_spi_ee93Cxx();
void test_spi_ee25xxx();
void test_spi_adis16350();


int main(void)
{
	void initPio();
	void initSpi();

	enum { MODE_SPI_EE93Cxx = 0, MODE_SPI_EE25xxx, MODE_SPI_ADIS16350 };

	const uint8_t mode = MODE_SPI_ADIS16350;

	cli();
	initSystem();
	initPio();
	switch (mode)
	{
	case MODE_SPI_EE93Cxx:
		ee93Cxx_init(1);
		break;
	case MODE_SPI_EE25xxx:
		ee25xxx_init(1);
		break;
	case MODE_SPI_ADIS16350:
		adis16350_init(1);
		break;
	}
	sei();

	switch (mode)
	{
	case MODE_SPI_EE93Cxx:
		ee93Cxx_chip_deselect();
		test_spi_ee93Cxx();
		break;
	case MODE_SPI_EE25xxx:
		ee25xxx_chip_deselect();
		test_spi_ee25xxx();
		break;
	case MODE_SPI_ADIS16350:
		adis16350_chip_deselect();
		break;
	}


	for (;;)
	{
		switch (mode)
		{
		case MODE_SPI_EE93Cxx:
		case MODE_SPI_EE25xxx:
			sleep_mode(); 
			break;
		case MODE_SPI_ADIS16350:
			test_spi_adis16350();
			break;
		}
	}

	return 0;
}

void initSystem()
{
	// Analog Comparator
	ACSR &= ~(_BV(ACIE));  // analog comparator interrupt disable
	ACSR |= _BV(ACD);  // analog comparator disable
}

void test_spi_ee93Cxx()
{
}

void test_spi_ee25xxx()
{
/*
 * FIXME [check] >> do not complete test yet. so need to check
 */

	int ret;
	uint8_t byte = 0x00;

	//
	PORTA = 0xFF;
	_delay_ms(500);
	PORTA = 0x00;
	_delay_ms(500);

	//
	while (!ee25xxx_is_ready()) ;
	_delay_ms(100);

	ret = ee25xxx_read_status_register(&byte);
	if (0 != ret)
	{
		PORTA = byte;
		_delay_ms(2000);
	}
	else
	{
		PORTA = 0xF8;
		return;
	}

	//
	//PORTA = 0x4F;
	//_delay_ms(500);

	//
	while (!ee25xxx_is_ready()) ;
	_delay_ms(100);

	ret = ee25xxx_set_write_enable();
	if (0 == ret)
	{
		PORTA = 0xF4;
		return;
	}

	//
	PORTA = 0xFF;
	_delay_ms(500);
	PORTA = 0x00;
	_delay_ms(500);

	//
	while (!ee25xxx_is_ready()) ;
	_delay_ms(100);

	byte = 0x00;
	ret = ee25xxx_read_status_register(&byte);
	if (0 != ret)
	{
		PORTA = byte;
		_delay_ms(2000);
	}
	else
	{
		PORTA = 0xF2;
		return;
	}

	//
	PORTA = 0xFF;
	_delay_ms(500);
	PORTA = 0x00;
	_delay_ms(500);
/*
	PORTA = 0xC0;
	_delay_ms(500);
	PORTA = 0x00;

	const uint16_t len = 64;
	uint8_t buf[len];
	for (uint16_t i = 0; i < len; ++i)
	{
		//buf[i] = i + 1;
		buf[i] = len - i;
		//buf[i] = (i + 1) * 2;
	}
	
	const uint16_t eeaddr = 0x0080;
	int ret;

	// write a byte
	for (uint16_t i = 0; i < len; ++i)
	{
		PORTB &= ~(_BV(PB0));  // chip select
		ret = ee25xxx_set_write_enable();
		PORTB |= _BV(PB0);  // chip deselect
		_delay_ms(20);  // necessary: delay for twr time

		PORTB &= ~(_BV(PB0));  // chip select
		ret = ee93Cxx_write_a_byte(eeaddr + i, buf[i]);
		PORTB |= _BV(PB0);  // chip deselect
		_delay_ms(20);  // necessary: delay for twr time

		//if (ret) PORTA = buf[i];
		//else PORTA = ret & 0x00FF;
		//_delay_ms(500);
	}

	//
	PORTA = 0xC0;
	_delay_ms(1000);
	PORTA = 0x00;

	// read a byte
	uint8_t byte = 0x00;
	for (uint16_t i = 0; i < len; ++i)
	{
		byte = 0x00;
		PORTB &= ~(_BV(PB0));  // chip select
		ret = ee93Cxx_read_a_byte(eeaddr + i, &byte);
		PORTB |= _BV(PB0);  // chip deselect

		if (ret) PORTA = byte;
		else PORTA = ret & 0x00FF;
		_delay_ms(500);
	}

	//
	PORTA = 0xC0;
	_delay_ms(500);
*/
}

void test_spi_adis16350()
{
	int ret;
	uint16_t word = 0x0000;

	//
	ret = adis16350_read_a_register(ADIS16350_XGYRO_OUT, &word);
	if (0 != ret)
	{
		PORTA = (uint8_t)((word >> 8) & 0x00FF);
		_delay_ms(500);
		PORTA = (uint8_t)(word & 0x00FF);
		_delay_ms(500);
	}
	else
	{
		PORTA = 0xCC;
		_delay_ms(500);
	}

	PORTA = 0x00;
	_delay_ms(10);
}
