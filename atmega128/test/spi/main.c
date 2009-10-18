#include "spi_eeprom.h"
#include "spi_adis16350.h"
#include "usart.h"
#include <avr/sleep.h>
#include <avr/interrupt.h>
#include <util/delay.h>


void test_spi_ee93Cxx();
void test_spi_ee25xxx();
void test_spi_adis16350();
void test_spi_adis16350_self_test();


void system_init()
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
	DDRA = 0xFF;  // uses all pins on PortA for output
	//DDRB = 0xFF;  // uses all pins on PortB for SPI
	DDRC = 0xFF;  // uses all pins on PortC for output
	DDRD = 0xFF;  // uses all pins on PortD for output
}

int main()
{
	enum { MODE_SPI_EE93Cxx = 0, MODE_SPI_EE25xxx, MODE_SPI_ADIS16350, MODE_SPI_ADIS16350_SELF_TEST } mode = MODE_SPI_ADIS16350;

	cli();
	system_init();
	switch (mode)
	{
	case MODE_SPI_EE93Cxx:
		ee93Cxx_init(1);
		break;
	case MODE_SPI_EE25xxx:
		ee25xxx_init(1);
		break;
	case MODE_SPI_ADIS16350:
		usart0_init(57600UL);
	case MODE_SPI_ADIS16350_SELF_TEST:
		adis16350_init(1);
		break;
	}
	sei();

	switch (mode)
	{
	case MODE_SPI_EE93Cxx:
		//ee93Cxx_chip_deselect();
		test_spi_ee93Cxx();
		break;
	case MODE_SPI_EE25xxx:
		//ee25xxx_chip_deselect();
		test_spi_ee25xxx();
		break;
	case MODE_SPI_ADIS16350:
		//adis16350_chip_deselect();
		break;
	case MODE_SPI_ADIS16350_SELF_TEST:
		//adis16350_chip_deselect();
		test_spi_adis16350_self_test();
		break;
	}

	while (1)
	{
		switch (mode)
		{
		case MODE_SPI_EE93Cxx:
		case MODE_SPI_EE25xxx:
		case MODE_SPI_ADIS16350_SELF_TEST:
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

	for (int k = 0; k <= 3; ++k)
	{
		//
		const uint16_t eeaddr = 0x0080;
		//const uint16_t eeaddr = 0x0145;
		//const uint16_t eeaddr = 0x2121;
		const uint16_t len = 64;
		uint8_t buf[len];

		for (uint16_t i = 0; i < len; ++i)
		{
			if (0 == k) buf[i] = i + 1;
			else if (1 == k) buf[i] = len - i;
			else if (2 == k) buf[i] = (i + 1) * 2;
			else buf[i] = 0;
		}
	
		// write bytes
		for (uint16_t i = 0; i < len; ++i)
		{
			// caution: write-enable process is necessary
			//	after write-data process, write-enable bit come to be reset
			//	so before write-data process, it is necessary to make write-enable bit on
			while (!ee25xxx_is_ready()) ;
			ret = ee25xxx_set_write_enable();

			while (!ee25xxx_is_ready()) ;
			ret = ee25xxx_write_a_byte(eeaddr + i, buf[i]);
	/*
			if (ret) PORTA = buf[i];
			_delay_ms(50);
			PORTA = 0x00;
			_delay_ms(50);
	*/
		}

		// read bytes
		for (uint16_t i = 0; i < len; ++i)
		{
			uint8_t byte = 0x00;
			while (!ee25xxx_is_ready()) ;
			ret = ee25xxx_read_a_byte(eeaddr + i, &byte);

			if (ret) PORTA = byte;
			_delay_ms(200);
			PORTA = 0x00;
			_delay_ms(10);
		}

		//
		PORTA = 0xFF;
		_delay_ms(500);
		PORTA = 0x00;
		_delay_ms(500);
	}
}

void test_spi_adis16350()
{
	int ret;
	uint16_t word = 0x0000;
	ret = adis16350_read_a_register(ADIS16350_ZGYRO_OUT, &word);
	if (0 != ret)
	{
		const uint8_t nd_flag = ((word >> 15) & 0x01) == 0x01;  // new data indicator
		const uint8_t ea_flag = ((word >> 14) & 0x01) == 0x01;  // system error or alarm condition
		const uint16_t data = word & 0x3FFF;  // output data

		//PORTA = (uint8_t)((data >> 8) & 0x00FF);
		//PORTA = (uint8_t)(data & 0x00FF);
		//_delay_ms(10);

		usart0_push_char('$');  // stx
		usart0_push_char(nd_flag ? 'y' : 'n');
		usart0_push_char(ea_flag ? 'y' : 'n');
		usart0_push_char(hex2ascii((data >> 12) & 0x0F));
		usart0_push_char(hex2ascii((data >> 8) & 0x0F));
		usart0_push_char(hex2ascii((data >> 4) & 0x0F));
		usart0_push_char(hex2ascii(data & 0x0F));
		usart0_push_char('&');  // etx
	}
	else
	{
		PORTA = 0xCC;
		_delay_ms(500);
	}

	PORTA = 0x00;
	_delay_ms(500);
}

void test_spi_adis16350_self_test()
{
	uint16_t word = 0x0000;
	int ret;

	//
	PORTA = 0xFF;
	_delay_ms(500);
	PORTA = 0x00;
	_delay_ms(500);
/*
	ret = adis16350_write_a_register(ADIS16350_MSC_CTRL, 0x0001);
	_delay_ms(50);

	// manual flash update command to store the nonvolatile data registers
	ret = adis16350_write_a_register(ADIS16350_COMMAND, 0x0001 << 3);
	_delay_ms(100);

	//
	ret = adis16350_read_a_register(ADIS16350_MSC_CTRL, &word);

	PORTA = (uint8_t)((word >> 8) & 0x00FF);
	_delay_ms(500);
	PORTA = (uint8_t)(word & 0x00FF);
	_delay_ms(500);

	//
	word |= 0x01 << 9;
	ret = adis16350_write_a_register(ADIS16350_MSC_CTRL, word);

	//
	ret = adis16350_read_a_register(ADIS16350_STATUS, &word);

	PORTA = (uint8_t)((word >> 4) & 0x01);
	_delay_ms(500);
*/
	while (1)
	{
		PORTA = 0xFF;
		_delay_ms(200);
		PORTA = 0x00;
		_delay_ms(200);

		const int test_mode = 3;
		if (1 == test_mode)
		{
			ret = adis16350_read_a_register(ADIS16350_SMPL_PRD, &word);
			PORTA = (uint8_t)(word & 0xFF);
			_delay_ms(500);
		}
		else if (2 == test_mode)
		{
			ret = adis16350_read_a_register(ADIS16350_SENS_AVG, &word);
			PORTA = (uint8_t)((word >> 8) & 0x07);
			_delay_ms(500);
			PORTA = (uint8_t)(word & 0x07);
			_delay_ms(500);
		}
		else if (3 == test_mode)
		{
			adis16350_write_a_register(ADIS16350_GPIO_CTRL, 0x0201);
			_delay_ms(500);

			ret = adis16350_read_a_register(ADIS16350_GPIO_CTRL, &word);
			_delay_ms(100);

			ret = adis16350_read_a_register(ADIS16350_GPIO_CTRL, &word);
			PORTA = (uint8_t)((word >> 8) & 0x03);
			_delay_ms(500);
			PORTA = (uint8_t)(word & 0x03);
			_delay_ms(500);
		}
	}

	//
	PORTA = 0xFF;
	_delay_ms(500);
	PORTA = 0x00;
	_delay_ms(500);
}
