#include "i2c_rtc.h"
#include "i2c.h"
#include <util/twi.h>
#include <util/delay.h>


//-----------------------------------------------------------------------------
//

int ds1307_write_a_byte(const uint8_t addr, const uint8_t byte);
int ds1307_write_bytes(const uint8_t addr, const uint16_t bufLen, const uint8_t *buf, uint16_t *byteLenWritten);
int ds1307_read_a_byte(const uint8_t addr, uint8_t *byte);
int ds1307_read_bytes(const uint8_t addr, const uint16_t bufLen, uint8_t *buf, uint16_t *byteLenRead);

int ds1307_init()
{
	uint8_t val;
	if (0 == ds1307_read_a_byte(0x02, &val))
		return 0;
	if (val & 0x40)  // 12hr selected ?
	{
		const int ret = ds1307_write_a_byte(0x02, val & 0x3F);  // 24hr
		_delay_us(50);
		if (0 == ret)
			return 0;
	}

	//
	if (0 == ds1307_read_a_byte(0x00, &val))
		return 0;
	if (val & 0x80)  // Clock running ?
	{
		const int ret = ds1307_write_a_byte(0x00, val & 0x7F);
		_delay_us(50);
		if (0 == ret)
			return 0;
	}

	ds1307_write_a_byte(0x07, 0x90);  // Enable 1Hz SQW signal.

	return 1;
}

int ds1307_set_date_time(const uint8_t year, const uint8_t month, const uint8_t date, const uint8_t day_of_week, const uint8_t hour, const uint8_t minute, const uint8_t second)
{
	const uint8_t register_addr = 0x00;
	const uint16_t bufLen = 7;
	const uint8_t buf[] =
	{
		(((second / 10) << 4) | (second % 10)) & 0x7F,  // Seconds to zero. start clock also.
		((minute / 10) << 4) | (minute % 10),
		((hour / 10) << 4) | (hour % 10),
		day_of_week,
		((date / 10) << 4) | (date % 10),
		((month / 10) << 4) | (month % 10),
		((year / 10) << 4) | (year % 10)
	};

	uint16_t byteLenWritten = 0;
	const int ret = ds1307_write_bytes(register_addr, bufLen, buf, &byteLenWritten);
	_delay_us(50);

	return bufLen <= byteLenWritten ? ret : 0;
}

int ds1307_get_time(uint8_t *hour, uint8_t *minute, uint8_t *second)
{
	const uint8_t register_addr = 0x00;
	const uint16_t bufLen = 3;
	uint8_t buf[bufLen];

	uint16_t byteLenRead = 0;
	const int ret = ds1307_read_bytes(register_addr, bufLen, buf, &byteLenRead);
	if (ret && bufLen == byteLenRead)
	{
		*second = (buf[0] & 0x0F) + (((buf[0] >> 4) & 0x07) * 10);
		*minute = (buf[1] & 0x0F) + (((buf[1] >> 4) & 0x07) * 10);
		*hour = (buf[2] & 0x0F) + (((buf[2] >> 4) & 0x03) * 10);
		return ret;
	}
	else return 0;
}

int ds1307_get_date(uint8_t *year, uint8_t *month, uint8_t *date, uint8_t *day_of_week)
{
	const uint8_t register_addr = 0x03;
	const uint16_t bufLen = 4;
	uint8_t buf[bufLen];

	uint16_t byteLenRead = 0;
	const int ret = ds1307_read_bytes(register_addr, bufLen, buf, &byteLenRead);
	if (ret && bufLen == byteLenRead)
	{
		*day_of_week = (buf[0] & 0x07);
		*date = (buf[1] & 0x0F) + (((buf[1] >> 4) & 0x03) * 10);
		*month = (buf[2] & 0x0F) + (((buf[2] >> 4) & 0x01) * 10);
		*year = (buf[3] & 0x0F) + (((buf[3] >> 4) & 0x0F) * 10);
		return ret;
	}
	else return 0;
}

//-----------------------------------------------------------------------------
//

/*
 * Maximal number of iterations to wait for a device to respond for a
 * selection.  Should be large enough to allow for a pending write to
 * complete, but low enough to properly abort an infinite loop in case
 * a slave is broken or not present at all.  With 100 kHz TWI clock,
 * transfering the start condition and SLA+R/W packet takes about 10
 * µs.  The longest write period is supposed to not exceed ~ 10 ms.
 * Thus, normal operation should not require more than 100 iterations
 * to get the device to respond to a selection.
 */
static const uint8_t MAX_RESTART_ITER_COUNT = 200;

/*
 * TWI address for DS1307.
 */
static const uint8_t TWI_SLA_DS1307 = 0xD0;


int ds1307_write_a_byte(const uint8_t addr, const uint8_t byte)
{
	// Patch DS1307 address into SLA.
	const uint8_t sla = TWI_SLA_DS1307;

	uint8_t iter = 0;
	I2C_STATUS status = I2C_ERR_RESTART;

	while (I2C_ERR_RESTART == status || I2C_ERR_ARB_LOST == status)
	{
		/*
		 * First cycle: Master transmitter mode.
		 */

		if (I2C_ERR_RESTART == status && iter++ >= MAX_RESTART_ITER_COUNT)
			return 0;

		// Send start condition.
		status = i2c_start();
		if (I2C_ERR_RESTART == status || I2C_ERR_ARB_LOST == status)
			continue;
		else if (I2C_ERR_QUIT_WITHOUT_STOP == status)
			return 0;

		// Send SLA+W.
		if (I2C_OK == status)
		{
			status = i2c_sla_w(sla);
			if (I2C_ERR_RESTART == status || I2C_ERR_ARB_LOST == status)
				continue;
			//else if (I2C_ERR_QUIT_WITH_STOP == status) ;
		}

		// 8-bit register address.
		if (I2C_OK == status)
		{
			status = i2c_address(addr);
			if (I2C_ERR_ARB_LOST == status)
				continue;
			//else if (I2C_QUIT == status || I2C_ERR_QUIT_WITH_STOP == status) ;
		}

		// Transmit a byte.
		if (I2C_OK == status)
		{
			status = i2c_master_write_a_byte(byte);
			if (I2C_ERR_ARB_LOST == status)
				continue;
			//else if (I2C_ERR_QUIT_WITH_STOP == status) ;
		}

		// Send stop condition.
		if (I2C_OK == status || I2C_QUIT == status || I2C_ERR_QUIT_WITH_STOP == status)
		{
			i2c_stop();
			return I2C_ERR_QUIT_WITH_STOP == status ? 0 : 1;
		}
	}

	return 1;
}

int ds1307_write_bytes(const uint8_t addr, const uint16_t bufLen, const uint8_t *buf, uint16_t *byteLenWritten)
{
	// Patch DS1307 address into SLA.
	const uint8_t sla = TWI_SLA_DS1307;

	uint8_t iter = 0;
	I2C_STATUS status = I2C_ERR_RESTART;

	while (I2C_ERR_RESTART == status || I2C_ERR_ARB_LOST == status)
	{
		/*
		 * First cycle: Master transmitter mode.
		 */

		if (I2C_ERR_RESTART == status && iter++ >= MAX_RESTART_ITER_COUNT)
			return 0;

		// Send start condition.
		status = i2c_start();
		if (I2C_ERR_RESTART == status || I2C_ERR_ARB_LOST == status)
			continue;
		else if (I2C_ERR_QUIT_WITHOUT_STOP == status)
			return 0;

		// Send SLA+W.
		if (I2C_OK == status)
		{
			status = i2c_sla_w(sla);
			if (I2C_ERR_RESTART == status || I2C_ERR_ARB_LOST == status)
				continue;
			//else if (I2C_ERR_QUIT_WITH_STOP == status) ;
		}

		// 8-bit register address.
		if (I2C_OK == status)
		{
			status = i2c_address(addr);
			if (I2C_ERR_ARB_LOST == status)
				continue;
			//else if (I2C_QUIT == status || I2C_ERR_QUIT_WITH_STOP == status) ;
		}

		// Transmit a byte.
		if (I2C_OK == status)
		{
			status = i2c_master_write_bytes(bufLen, buf, byteLenWritten);
			if (I2C_ERR_ARB_LOST == status)
				continue;
			//else if (I2C_ERR_QUIT_WITH_STOP == status) ;
		}

		// Send stop condition.
		if (I2C_OK == status || I2C_QUIT == status || I2C_ERR_QUIT_WITH_STOP == status)
		{
			i2c_stop();
			return I2C_ERR_QUIT_WITH_STOP == status ? 0 : 1;
		}
	}

	return 1;
}

int ds1307_read_a_byte(const uint8_t addr, uint8_t *byte)
{
	// Patch DS1307 address into SLA.
	const uint8_t sla = TWI_SLA_DS1307;

	/*
	 * First cycle: Master transmitter mode.
	 */
	uint8_t iter = 0;
	I2C_STATUS status = I2C_ERR_RESTART;

	while (I2C_ERR_RESTART == status || I2C_ERR_ARB_LOST == status)
	{
		if (I2C_ERR_RESTART == status && iter++ >= MAX_RESTART_ITER_COUNT)
			return 0;

		// Send start condition.
		status = i2c_start();
		if (I2C_ERR_RESTART == status || I2C_ERR_ARB_LOST == status)
			continue;
		else if (I2C_ERR_QUIT_WITHOUT_STOP == status)
			return 0;

		// Send SLA+W.
		if (I2C_OK == status)
		{
			status = i2c_sla_w(sla);
			if (I2C_ERR_RESTART == status || I2C_ERR_ARB_LOST == status)
				continue;
			//else if (I2C_ERR_QUIT_WITH_STOP == status) ;
		}

		// 8-bit register address.
		if (I2C_OK == status)
		{
			status = i2c_address(addr);
			if (I2C_ERR_ARB_LOST == status)
				continue;
			//else if (I2C_QUIT == status || I2C_ERR_QUIT_WITH_STOP == status) ;
		}

		// Send stop condition.
		if (I2C_OK == status || I2C_QUIT == status || I2C_ERR_QUIT_WITH_STOP == status)
		{
			i2c_stop();
			if (I2C_OK == status) break;
			else return I2C_ERR_QUIT_WITH_STOP == status ? 0 : 1;
		}
	}

	//
	_delay_us(50);

	/*
	 * Next cycle(s): Master receiver mode.
	 */
	iter = 0;
	status = I2C_ERR_RESTART;

	while (I2C_ERR_RESTART == status || I2C_ERR_ARB_LOST == status)
	{
		if (I2C_ERR_RESTART == status && iter++ >= MAX_RESTART_ITER_COUNT)
			return 0;

		// Send start condition.
		status = i2c_start();
		if (I2C_ERR_RESTART == status || I2C_ERR_ARB_LOST == status)
			continue;
		else if (I2C_ERR_QUIT_WITHOUT_STOP == status)
			return 0;

		// Send SLA+R.
		if (I2C_OK == status)
		{
			status = i2c_sla_r(sla);
			if (I2C_ERR_ARB_LOST == status)
				continue;
			//else if (I2C_QUIT == status || I2C_ERR_QUIT_WITH_STOP == status) ;
		}

		// Receive a byte.
		if (I2C_OK == status)
		{
			status = i2c_master_read_a_byte(byte);
			if (I2C_ERR_ARB_LOST == status)
				continue;
			//else if (I2C_ERR_QUIT_WITH_STOP == status) ;
		}

		// Send stop condition.
		if (I2C_OK == status || I2C_QUIT == status || I2C_ERR_QUIT_WITH_STOP == status)
		{
			i2c_stop();
			return I2C_ERR_QUIT_WITH_STOP == status ? 0 : 1;
		}
	}

	return 1;
}

int ds1307_read_bytes(const uint8_t addr, const uint16_t bufLen, uint8_t *buf, uint16_t *byteLenRead)
{
	// Patch DS1307 address into SLA.
	const uint8_t sla = TWI_SLA_DS1307;

	/*
	 * First cycle: Master transmitter mode.
	 */
	uint8_t iter = 0;
	I2C_STATUS status = I2C_ERR_RESTART;

	while (I2C_ERR_RESTART == status || I2C_ERR_ARB_LOST == status)
	{
		if (I2C_ERR_RESTART == status && iter++ >= MAX_RESTART_ITER_COUNT)
			return 0;

		// Send start condition.
		status = i2c_start();
		if (I2C_ERR_RESTART == status || I2C_ERR_ARB_LOST == status)
			continue;
		else if (I2C_ERR_QUIT_WITHOUT_STOP == status)
			return 0;

		// Send SLA+W.
		if (I2C_OK == status)
		{
			status = i2c_sla_w(sla);
			if (I2C_ERR_RESTART == status || I2C_ERR_ARB_LOST == status)
				continue;
			//else if (I2C_ERR_QUIT_WITH_STOP == status) ;
		}

		// 8-bit register address.
		if (I2C_OK == status)
		{
			status = i2c_address(addr);
			if (I2C_ERR_ARB_LOST == status)
				continue;
			//else if (I2C_QUIT == status || I2C_ERR_QUIT_WITH_STOP == status) ;
		}

		// Send stop condition.
		if (I2C_OK == status || I2C_QUIT == status || I2C_ERR_QUIT_WITH_STOP == status)
		{
			i2c_stop();
			if (I2C_OK == status) break;
			else return I2C_ERR_QUIT_WITH_STOP == status ? 0 : 1;
		}
	}

	//
	_delay_us(50);

	/*
	 * Next cycle(s): Master receiver mode.
	 */
	iter = 0;
	status = I2C_ERR_RESTART;

	while (I2C_ERR_RESTART == status || I2C_ERR_ARB_LOST == status)
	{
		if (I2C_ERR_RESTART == status && iter++ >= MAX_RESTART_ITER_COUNT)
			return 0;

		// Send start condition.
		status = i2c_start();
		if (I2C_ERR_RESTART == status || I2C_ERR_ARB_LOST == status)
			continue;
		else if (I2C_ERR_QUIT_WITHOUT_STOP == status)
			return 0;

		// Send SLA+R.
		if (I2C_OK == status)
		{
			status = i2c_sla_r(sla);
			if (I2C_ERR_ARB_LOST == status)
				continue;
			//else if (I2C_QUIT == status || I2C_ERR_QUIT_WITH_STOP == status) ;
		}

		// Receive a byte.
		if (I2C_OK == status)
		{
			status = i2c_master_read_bytes(bufLen, buf, byteLenRead);
			if (I2C_ERR_ARB_LOST == status)
				continue;
			//else if (I2C_ERR_QUIT_WITH_STOP == status) ;
		}

		// Send stop condition.
		if (I2C_OK == status || I2C_QUIT == status || I2C_ERR_QUIT_WITH_STOP == status)
		{
			i2c_stop();
			return I2C_ERR_QUIT_WITH_STOP == status ? 0 : 1;
		}
	}

	return 1;
}

