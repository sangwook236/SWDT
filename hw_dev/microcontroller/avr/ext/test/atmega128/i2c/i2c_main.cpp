#include "i2c_eeprom.h"
#include "i2c_rtc.h"
#include "i2c_hmc6352.h"
#include <avr/sleep.h>
#include <avr/interrupt.h>
#include <util/delay.h>


namespace {
namespace local {

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
	// it makes port input register(PINn) high-impedance state that port output register(PORTn) outputs 0(low)
	// so that we can share the pin with other devices
	//PORTA = 0x00;
*/
	// uses all pins on PortD for output
	DDRD = 0xFF;

	// for 7-segment
	DDRA = 0xFF;
	DDRC = 0xFF;
	// for two yellow LEDs
	DDRG = 0x03;
}

void test_i2c_eeprom()
{
	PORTA = 0xC0;
	_delay_ms(500);

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

	// write case 1
	for (uint16_t i = 0; i < len; ++i)
	{
		ret = ee24Cxxx_write_a_byte(eeaddr + i, buf[i]);
		_delay_ms(20);  // necessary: delay for twr time

		//if (ret) PORTA = buf[i];
		//else PORTA = ret & 0x00FF;
		//_delay_ms(500);
	}
	// write case 2
/*
	uint16_t byteLenWritten = 0;
	ret = ee24Cxxx_write_bytes(eeaddr, len, buf, &byteLenWritten);  // error: Oops !!! i don't know why.
	_delay_ms(20);  // necessary: delay for twr time

	PORTA = ret & 0x00FF;
	_delay_ms(500);

	PORTA = byteLenWritten & 0x00FF;
	_delay_ms(500);
*/
	//
	PORTA = 0xC0;
	_delay_ms(1000);

	// read case 1
/*
	uint8_t byte = 0x00;
	for (uint16_t i = 0; i < len; ++i)
	{
		byte = 0x00;
		ret = ee24Cxxx_read_a_byte(eeaddr + i, &byte);

		if (ret) PORTA = byte;
		else PORTA = ret & 0x00FF;
		_delay_ms(500);
	}
*/
	// read case 2
	uint16_t byteLenRead = 0;
	uint8_t buf2[len];
	for (uint16_t i = 0; i < len; ++i) buf2[i] = 0x00;
	ret = ee24Cxxx_read_bytes(eeaddr, len, buf2, &byteLenRead);

	PORTA = byteLenRead & 0x00FF;
	_delay_ms(500);

	for (uint16_t i = 0; i < byteLenRead && i < len; ++i)
	{
		PORTA = buf2[i];
		_delay_ms(500);
	}

	//
	PORTA = 0xC0;
	_delay_ms(500);
}

void test_i2c_rtc()
{
	void four_digit_seven_segment_anode_commmon(const uint16_t four_digits);
	void four_digit_seven_segment_cathode_commmon(const uint16_t four_digits);

	// init ds1307
	if (0 == ds1307_init())
	{
		PORTA = 0x8F;
		_delay_ms(500);
		return;
	}

#if 0
	// set date & time: call just once after power-on
	if (0 == ds1307_set_date_time(9, 6, 26, 5, 20, 26, 0))
	{
		PORTA = 0x4F;
		_delay_ms(500);
		return;
	}
	else
	{
		PORTA = 0xFF;
		_delay_ms(500);
		return;
	}
#else
	//
	uint8_t year = 0, month = 0, date = 0, day_of_week = 0;
	uint8_t hour = 0, minute = 0, second = 0;
	int display_time = 1;
	int i = 0;
	while (1)
	{
		if (0 != display_time)
		{
			// get time
			if (0 == ds1307_get_time(&hour, &minute, &second))
			{
				PORTA = 0x1F;
				_delay_ms(500);
				return;
			}

			//four_digit_seven_segment_anode_commmon(hour * 100 + minute);
			four_digit_seven_segment_cathode_commmon(hour * 100 + minute);
		}
		else
		{
			// get date
			if (0 == ds1307_get_date(&year, &month, &date, &day_of_week))
			{
				PORTA = 0x2F;
				_delay_ms(500);
				return;
			}

			//four_digit_seven_segment_anode_commmon(month * 100 + date);
			four_digit_seven_segment_cathode_commmon(month * 100 + date);
		}

		if (200 == ++i) 
		{
			i = 0;
			display_time = !display_time;
		}

		_delay_ms(1);
	}
#endif
}

void test_i2c_hmc6352()
{
	void four_digit_seven_segment_anode_commmon(const uint16_t four_digits);
	void four_digit_seven_segment_cathode_commmon(const uint16_t four_digits);

	PORTA = 0xFF;
	_delay_ms(100);
	PORTA = 0x00;
	_delay_ms(10);

#if 0
	//
	// operational mode
	//
	uint8_t op_mode = 0x00;
	hmc6352_read_from_ram(HMC6352_RAM__OPERATIONAL_MODE, &op_mode);

	//op_mode = (op_mode & 0xFC) | 0x00;  // stand-by mode
	op_mode = (op_mode & 0xFC) | 0x01;  // query mode
	//op_mode = (op_mode & 0xFC) | 0x02;  // continuous mode
	hmc6352_write_to_ram(HMC6352_RAM__OPERATIONAL_MODE, op_mode);
	_delay_ms(10);
	op_mode = 0x00;
	hmc6352_read_from_ram(HMC6352_RAM__OPERATIONAL_MODE, &op_mode);
	PORTA = op_mode;
	_delay_ms(1000);

	op_mode = (op_mode & 0xFC) | 0x00;  // stand-by mode
	//op_mode = (op_mode & 0xFC) | 0x01;  // query mode
	//op_mode = (op_mode & 0xFC) | 0x02;  // continuous mode
	hmc6352_write_to_ram(HMC6352_RAM__OPERATIONAL_MODE, op_mode);
	_delay_ms(10);
	op_mode = 0x00;
	hmc6352_read_from_ram(HMC6352_RAM__OPERATIONAL_MODE, &op_mode);
	PORTA = op_mode;
	_delay_ms(1000);

	//
	// output data mode
	//
	uint8_t out_mode = 0x00;
	hmc6352_read_from_ram(HMC6352_RAM__OUTPUT_DATA_MODE, &out_mode);
	PORTA = out_mode;
	_delay_ms(1000);

	PORTA = 0xFF;
	_delay_ms(100);
	PORTA = 0x00;
	_delay_ms(10);
#endif

	uint8_t data[2] = { 0x00, 0x00 };

	enum { MODE_STAND_BY, MODE_QUERY, MODE_CONTINUOUS } mode = MODE_STAND_BY;

	uint8_t op_mode = 0x00;
	hmc6352_read_from_ram(HMC6352_RAM__OPERATIONAL_MODE, &op_mode);
	switch (mode)
	{
	case MODE_STAND_BY:
		op_mode = (op_mode & 0xFC) | 0x00;  // stand-by mode
		hmc6352_write_to_ram(HMC6352_RAM__OPERATIONAL_MODE, op_mode);
		_delay_ms(10);
		break;
	case MODE_QUERY:
		op_mode = (op_mode & 0xFC) | 0x01;  // query mode
		hmc6352_write_to_ram(HMC6352_RAM__OPERATIONAL_MODE, op_mode);
		_delay_ms(10);
		hmc6352_get_data(data);
		break;
	case MODE_CONTINUOUS:
		op_mode = (op_mode & 0xFC) | 0x02;  // continuous mode
		hmc6352_write_to_ram(HMC6352_RAM__OPERATIONAL_MODE, op_mode);
		_delay_ms(10);
		break;
	}

	uint16_t angle = 0x0000;
	int idx = 0;
	int flag = 1;
	while (1)
	{
		if (0 == idx)
		{
			switch (mode)
			{
			case MODE_STAND_BY:
				hmc6352_get_data(data);  // [0, 3599]
				break;
			case MODE_QUERY:
			case MODE_CONTINUOUS:
				hmc6352_read_word_data(data);  // [0, 3599]
				break;
			}

			//const uint8_t high = data[0];  // MSB
			//const uint8_t low = data[1];  // LSB
			angle = ((data[0] << 8) & 0xFF00) | (data[1] & 0x00FF);

			if (1 == flag)
			{
				PORTG =  0x01;
				flag = 0;
			}
			else
			{
				PORTG =  0x02;
				flag = 1;
			}
		}

		//four_digit_seven_segment_anode_commmon(angle);
		four_digit_seven_segment_cathode_commmon(angle);
		_delay_ms(5);

		idx = (idx + 1) % 10;
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_i2c {
}  // namespace my_i2c

int i2c_main()
{
	void i2c_init(const uint8_t is_fast_mode);

	enum { MODE_I2C_EEPROM = 0, MODE_I2C_RTC = 1, MODE_I2C_HMC6352 = 2 };

	//const uint8_t mode = MODE_I2C_EEPROM;
	//const uint8_t mode = MODE_I2C_RTC;
	const uint8_t mode = MODE_I2C_HMC6352;

	uint8_t is_fast_mode = 0;
	switch (mode)
	{
	case MODE_I2C_EEPROM:
	case MODE_I2C_HMC6352:
		is_fast_mode = 0;
		break;
	case MODE_I2C_RTC:
		is_fast_mode = 1;
		break;
	}

	cli();
	local::system_init();
	i2c_init(is_fast_mode);
	sei();

	//
	switch (mode)
	{
	case MODE_I2C_EEPROM:
		local::test_i2c_eeprom();
		break;
	case MODE_I2C_RTC:
		local::test_i2c_rtc();
		break;
	case MODE_I2C_HMC6352:
		local::test_i2c_hmc6352();
		break;
	}

	while (1)
		sleep_mode(); 

	return 0;
}
