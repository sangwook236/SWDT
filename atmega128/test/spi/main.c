#include "spi_eeprom.h"
#include "spi_adis16350.h"
#include <avr/sleep.h>
#include <avr/interrupt.h>
#include <util/delay.h>


void test_spi_ee93Cxx();
void test_spi_ee25xxx();
void test_spi_adis16350();


void initSystem()
{
	/*
	 *	analog comparator
	 */
	ACSR &= ~(_BV(ACIE));  // analog comparator interrupt disable
	ACSR |= _BV(ACD);  // analog comparator disable

	/*
	 *	I/O port
	 */
/*
	// uses all pins on PortA for input
	DDRA = 0x00;
	// it makes port input register(PINn) internally pulled-up state that port output register(PORTn) outputs 1(high)
	PORTA = 0xFF;
	// it makes port input register(PINn) high-impedence state that port output register(PORTn) outputs 0(low)
	// so that we can share the pin with other devices
	//PORTA = 0x00;
*/
	// uses all pins on PortB for SPI
	//DDRB = 0xFF;

	// uses all pins on PortA for output
	DDRA = 0xFF;

	// uses all pins on PortC for output
	DDRC = 0xFF;
}

int main()
{
	enum { MODE_SPI_EE93Cxx = 0, MODE_SPI_EE25xxx, MODE_SPI_ADIS16350 };

	const uint8_t mode = MODE_SPI_ADIS16350;

	cli();
	initSystem();
	switch (mode)
	{
	case MODE_SPI_EE93Cxx:
		ee93Cxx_init();
		break;
	case MODE_SPI_EE25xxx:
		ee25xxx_init();
		break;
	case MODE_SPI_ADIS16350:
		adis16350_init();
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

void test_spi_ee93Cxx()
{
	// FIXME [add] >>
}

void test_spi_ee25xxx()
{
	int ret;
	uint8_t status = 0x00;

	//
	PORTA = 0xFF;
	_delay_ms(500);
	PORTA = 0x00;
	_delay_ms(500);

	//
	while (!ee25xxx_is_ready()) ;
	_delay_ms(10);

	ret = ee25xxx_read_status_register(&status);
	if (0 != ret)
	{
		PORTA = status;
		_delay_ms(1000);
	}
	else
	{
		PORTA = 0xF8;
		return;
	}

	// set write-disable
	while (!ee25xxx_is_ready()) ;

	ret = ee25xxx_set_write_disable();
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

	status = 0x00;
	ret = ee25xxx_read_status_register(&status);
	if (0 != ret)
	{
		PORTA = status;
		_delay_ms(1000);
	}
	else
	{
		PORTA = 0xF2;
		return;
	}

	// set write-enable
	while (!ee25xxx_is_ready()) ;

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

	status = 0x00;
	ret = ee25xxx_read_status_register(&status);
	if (0 != ret)
	{
		PORTA = status;
		_delay_ms(1000);
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

	//
	const uint16_t eeaddr = 0x0080;
	const uint16_t len = 64;
	uint8_t buf[len];

	for (uint16_t i = 0; i < len; ++i)
	{
		buf[i] = i + 1;
		//buf[i] = len - i;
		//buf[i] = (i + 1) * 2;
	}
	
	// write bytes
	for (uint16_t i = 0; i < len; ++i)
	{
		ret = ee25xxx_write_a_byte(eeaddr + i, buf[i]);
		while (!ee25xxx_is_ready()) ;

		// caution: write-enable process is necessary
		//	after write-data process, write-enable bit come to be reset
		ret = ee25xxx_set_write_enable();
		while (!ee25xxx_is_ready()) ;
/*
		if (ret) PORTA = buf[i];
		_delay_ms(50);
		PORTA = 0x00;
		_delay_ms(50);
*/
	}

	//
	PORTA = 0xFF;
	_delay_ms(500);
	PORTA = 0x00;
	_delay_ms(500);

	// read bytes
	for (uint16_t i = 0; i < len; ++i)
	{
		uint8_t byte = 0x00;
		ret = ee25xxx_read_a_byte(eeaddr + i, &byte);
		while (!ee25xxx_is_ready()) ;

		if (ret) PORTA = byte;
		_delay_ms(200);
		PORTA = 0x00;
		_delay_ms(110);
	}

	//
	PORTA = 0xFF;
	_delay_ms(500);
	PORTA = 0x00;
	_delay_ms(500);
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
