#include "i2c_eeprom.h"
#include "i2c.h"
#include <util/twi.h>


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
 * TWI address for 24Cxx EEPROM:
 *
 * 1 0 1 0 E2 E1 E0 R/~W	24C01/24C02
 * 1 0 1 0 E2 E1 A8 R/~W	24C04
 * 1 0 1 0 E2 A9 A8 R/~W	24C08
 * 1 0 1 0 A10 A9 A8 R/~W	24C16
 */
static const uint8_t TWI_SLA_24CXX = 0xA0;  // E2 E1 E0 = 0 0 0
/*
 * TWI address for 24Cxx EEPROM:
 *
 * 1 0 1 0 0 E1 E0 R/~W		24C128
 * 1 0 1 0 0 E1 E0 R/~W		24C256
 */
static const uint8_t TWI_SLA_24CXXX = 0xA0;  // E1 E0 = 0 0


int ee24Cxx_write_a_byte(const uint16_t eeaddr, const uint8_t byte)
{
	// patch high bits of EEPROM address into SLA
	const uint8_t sla = TWI_SLA_24CXX | ((eeaddr >> 7) & 0x0E);

	uint8_t iter = 0;
	I2C_STATUS status = I2C_ERR_RESTART;

	while (I2C_ERR_RESTART == status || I2C_ERR_ARB_LOST == status)
	{
		/*
		 * First cycle: master transmitter mode
		 */

		if (I2C_ERR_RESTART == status && iter++ >= MAX_RESTART_ITER_COUNT)
			return 0;

		// send start condition
		status = i2c_start();
		if (I2C_ERR_RESTART == status || I2C_ERR_ARB_LOST == status)
			continue;
		else if (I2C_ERR_QUIT_WITHOUT_STOP == status)
			return 0;

		// send SLA+W
		if (I2C_OK == status)
		{
			status = i2c_sla_w(sla);
			if (I2C_ERR_RESTART == status || I2C_ERR_ARB_LOST == status)
				continue;
			//else if (I2C_ERR_QUIT_WITH_STOP == status) ;
		}

		// low 8 bits of address
		if (I2C_OK == status)
		{
			status = i2c_address(eeaddr & 0x00FF);
			if (I2C_ERR_ARB_LOST == status)
				continue;
			//else if (I2C_QUIT == status || I2C_ERR_QUIT_WITH_STOP == status) ;
		}

		// transmit a byte
		if (I2C_OK == status)
		{
			status = i2c_master_write_a_byte(byte);
			if (I2C_ERR_ARB_LOST == status)
				continue;
			//else if (I2C_ERR_QUIT_WITH_STOP == status) ;
		}

		// send stop condition
		if (I2C_OK == status || I2C_QUIT == status || I2C_ERR_QUIT_WITH_STOP == status)
		{
			i2c_stop();
			return I2C_ERR_QUIT_WITH_STOP == status ? 0 : 1;
		}
	}

	return 1;
}

int ee24Cxx_write_a_page(const uint16_t eeaddr, const uint16_t bufLen, const uint8_t *buf, uint16_t *byteLenWritten)
{
	// 1K/2K: 8-byte page write
	// 4K/8K/16K: 16-byte page write
	const uint8_t PAGE_SIZE = 8;

	// patch high bits of EEPROM address into SLA
	const uint8_t sla = TWI_SLA_24CXX | ((eeaddr >> 7) & 0x0E);

	const uint16_t len = (eeaddr + bufLen < (eeaddr | (PAGE_SIZE - 1))) ? bufLen : (eeaddr | (PAGE_SIZE - 1)) + 1 - eeaddr;

	uint8_t iter = 0;
	I2C_STATUS status = I2C_ERR_RESTART;

	//*byteLenWritten = 0;
	while (I2C_ERR_RESTART == status || I2C_ERR_ARB_LOST == status)
	{
		/*
		 * First cycle: master transmitter mode
		 */

		if (I2C_ERR_RESTART == status && iter++ >= MAX_RESTART_ITER_COUNT)
			return 0;

		// send start condition
		status = i2c_start();
		if (I2C_ERR_RESTART == status || I2C_ERR_ARB_LOST == status)
			continue;
		else if (I2C_ERR_QUIT_WITHOUT_STOP == status)
			return 0;

		// send SLA+W
		if (I2C_OK == status)
		{
			status = i2c_sla_w(sla);
			if (I2C_ERR_RESTART == status || I2C_ERR_ARB_LOST == status)
				continue;
			//else if (I2C_ERR_QUIT_WITH_STOP == status) ;
		}

		// low 8 bits of address
		if (I2C_OK == status)
		{
			status = i2c_address(eeaddr & 0x00FF);
			if (I2C_ERR_ARB_LOST == status)
				continue;
			//else if (I2C_QUIT == status || I2C_ERR_QUIT_WITH_STOP == status) ;
		}

		// transmit bytes
		if (I2C_OK == status)
		{
			status = i2c_master_write_bytes(len, buf, byteLenWritten);
			if (I2C_ERR_ARB_LOST == status)
				continue;
			//else if (I2C_ERR_QUIT_WITH_STOP == status) ;
		}

		// send stop condition
		if (I2C_OK == status || I2C_QUIT == status || I2C_ERR_QUIT_WITH_STOP == status)
		{
			i2c_stop();
			return I2C_ERR_QUIT_WITH_STOP == status ? 0 : 1;
		}
	}

	return 1;
}

int ee24Cxx_write_bytes(const uint16_t eeaddr, const uint16_t bufLen, const uint8_t *buf, uint16_t *byteLenWritten)
{
	uint16_t len = 0;

	uint16_t eeaddr2 = eeaddr;
	uint16_t bufLen2 = bufLen;
	do
	{
		len = 0;
		const int ret = ee24Cxx_write_a_page(eeaddr2, bufLen2, buf, &len);
		if (0 == ret) return 0;

		eeaddr2 += len;
		bufLen2 -= len;
		buf += len;
		*byteLenWritten += len;
	} while (bufLen2 > 0);

	return 1;
}

int ee24Cxx_read_a_byte(const uint16_t eeaddr, uint8_t *byte)
{
	// patch high bits of EEPROM address into SLA
	const uint8_t sla = TWI_SLA_24CXX | ((eeaddr >> 7) & 0x0E);

	uint8_t iter = 0;
	I2C_STATUS status = I2C_ERR_RESTART;

	while (I2C_ERR_RESTART == status || I2C_ERR_ARB_LOST == status)
	{
		/*
		 * First cycle: master transmitter mode
		 */

		if (I2C_ERR_RESTART == status && iter++ >= MAX_RESTART_ITER_COUNT)
			return 0;

		// send start condition
		status = i2c_start();
		if (I2C_ERR_RESTART == status || I2C_ERR_ARB_LOST == status)
			continue;
		else if (I2C_ERR_QUIT_WITHOUT_STOP == status)
			return 0;

		// send SLA+W
		if (I2C_OK == status)
		{
			status = i2c_sla_w(sla);
			if (I2C_ERR_RESTART == status || I2C_ERR_ARB_LOST == status)
				continue;
			//else if (I2C_ERR_QUIT_WITH_STOP == status) ;
		}

		// low 8 bits of address
		if (I2C_OK == status)
		{
			status = i2c_address(eeaddr & 0x00FF);
			if (I2C_ERR_ARB_LOST == status)
				continue;
			//else if (I2C_QUIT == status || I2C_ERR_QUIT_WITH_STOP == status) ;
		}

		/*
		 * Next cycle(s): master receiver mode
		 */

		// send (rep.) start condition
		if (I2C_OK == status)
		{
			status = i2c_repeated_start();
			if (I2C_ERR_RESTART == status || I2C_ERR_ARB_LOST == status)
				continue;
			//else if (I2C_ERR_QUIT_WITH_STOP == status) ;
		}

		// send SLA+R
		if (I2C_OK == status)
		{
			status = i2c_sla_r(sla);
			if (I2C_ERR_ARB_LOST == status)
				continue;
			//else if (I2C_QUIT == status || I2C_ERR_QUIT_WITH_STOP == status) ;
		}

		// receive a byte
		if (I2C_OK == status)
		{
			status = i2c_master_read_a_byte(byte);
			if (I2C_ERR_ARB_LOST == status)
				continue;
			//else if (I2C_ERR_QUIT_WITH_STOP == status) ;
		}

		// send stop condition
		if (I2C_OK == status || I2C_QUIT == status || I2C_ERR_QUIT_WITH_STOP == status)
		{
			i2c_stop();
			return I2C_ERR_QUIT_WITH_STOP == status ? 0 : 1;
		}
	}

	return 1;
}

int ee24Cxx_read_bytes(const uint16_t eeaddr, const uint16_t bufLen, uint8_t *buf, uint16_t *byteLenRead)
{
	// patch high bits of EEPROM address into SLA
	const uint8_t sla = TWI_SLA_24CXX | ((eeaddr >> 7) & 0x0E);

	uint8_t iter = 0;
	I2C_STATUS status = I2C_ERR_RESTART;

	//*byteLenRead = 0;
	while (I2C_ERR_RESTART == status || I2C_ERR_ARB_LOST == status)
	{
		/*
		 * First cycle: master transmitter mode
		 */

		if (I2C_ERR_RESTART == status && iter++ >= MAX_RESTART_ITER_COUNT)
			return 0;

		// send start condition
		status = i2c_start();
		if (I2C_ERR_RESTART == status || I2C_ERR_ARB_LOST == status)
			continue;
		else if (I2C_ERR_QUIT_WITHOUT_STOP == status)
			return 0;

		// send SLA+W
		if (I2C_OK == status)
		{
			status = i2c_sla_w(sla);
			if (I2C_ERR_RESTART == status || I2C_ERR_ARB_LOST == status)
				continue;
			//else if (I2C_ERR_QUIT_WITH_STOP == status) ;
		}

		// low 8 bits of address
		if (I2C_OK == status)
		{
			status = i2c_address(eeaddr & 0x00FF);
			if (I2C_ERR_ARB_LOST == status)
				continue;
			//else if (I2C_QUIT == status || I2C_ERR_QUIT_WITH_STOP == status) ;
		}

		/*
		 * Next cycle(s): master receiver mode
		 */

		// send (rep.) start condition
		if (I2C_OK == status)
		{
			status = i2c_repeated_start();
			if (I2C_ERR_RESTART == status || I2C_ERR_ARB_LOST == status)
				continue;
			//else if (I2C_ERR_QUIT_WITH_STOP == status) ;
		}

		// send SLA+R
		if (I2C_OK == status)
		{
			status = i2c_sla_r(sla);
			if (I2C_ERR_ARB_LOST == status)
				continue;
			//else if (I2C_QUIT == status || I2C_ERR_QUIT_WITH_STOP == status) ;
		}

		// receive bytes
		if (I2C_OK == status)
		{
			status = i2c_master_read_bytes(bufLen, buf, byteLenRead);
			if (I2C_ERR_ARB_LOST == status)
				continue;
			//else if (I2C_ERR_QUIT_WITH_STOP == status) ;
		}

		// send stop condition
		if (I2C_OK == status || I2C_QUIT == status || I2C_ERR_QUIT_WITH_STOP == status)
		{
			i2c_stop();
			return I2C_ERR_QUIT_WITH_STOP == status ? 0 : 1;
		}
	}

	return 1;
}

//-----------------------------------------------------------------------------
//

int ee24Cxxx_write_a_byte(const uint16_t eeaddr, const uint8_t byte)
{
	// patch hardwired input pins of EEPROM address into SLA
	const uint8_t sla = TWI_SLA_24CXXX;

	uint8_t iter = 0;
	I2C_STATUS status = I2C_ERR_RESTART;

	while (I2C_ERR_RESTART == status || I2C_ERR_ARB_LOST == status)
	{
		/*
		 * First cycle: master transmitter mode
		 */

		if (I2C_ERR_RESTART == status && iter++ >= MAX_RESTART_ITER_COUNT)
			return 0;

		// send start condition
		status = i2c_start();
		if (I2C_ERR_RESTART == status || I2C_ERR_ARB_LOST == status)
			continue;
		else if (I2C_ERR_QUIT_WITHOUT_STOP == status)
			return 0;

		// send SLA+W
		if (I2C_OK == status)
		{
			status = i2c_sla_w(sla);
			if (I2C_ERR_RESTART == status || I2C_ERR_ARB_LOST == status)
				continue;
			//else if (I2C_ERR_QUIT_WITH_STOP == status) ;
		}

		// first 8 bits of address
		if (I2C_OK == status)
		{
			status = i2c_address((eeaddr >> 8) & 0x00FF);
			if (I2C_ERR_ARB_LOST == status)
				continue;
			//else if (I2C_QUIT == status || I2C_ERR_QUIT_WITH_STOP == status) ;
		}

		// second 8 bits of address
		if (I2C_OK == status)
		{
			status = i2c_address(eeaddr & 0x00FF);
			if (I2C_ERR_ARB_LOST == status)
				continue;
			//else if (I2C_QUIT == status || I2C_ERR_QUIT_WITH_STOP == status) ;
		}

		// transmit bytes
		if (I2C_OK == status)
		{
			status = i2c_master_write_a_byte(byte);
			if (I2C_ERR_ARB_LOST == status)
				continue;
			//else if (I2C_ERR_QUIT_WITH_STOP == status) ;
		}

		// send stop condition
		if (I2C_OK == status || I2C_QUIT == status || I2C_ERR_QUIT_WITH_STOP == status)
		{
			i2c_stop();
			return I2C_ERR_QUIT_WITH_STOP == status ? 0 : 1;
		}
	}

	return 1;
}

int ee24Cxxx_write_a_page(const uint16_t eeaddr, const uint16_t bufLen, const uint8_t *buf, uint16_t *byteLenWritten)
{
	// 128K/256K: 64-byte page write
	const uint8_t PAGE_SIZE = 64;

	// patch hardwired input pins of EEPROM address into SLA
	const uint8_t sla = TWI_SLA_24CXXX;

	const uint16_t len = (eeaddr + bufLen < (eeaddr | (PAGE_SIZE - 1))) ? bufLen : (eeaddr | (PAGE_SIZE - 1)) + 1 - eeaddr;

	uint8_t iter = 0;
	I2C_STATUS status = I2C_ERR_RESTART;

	//*byteLenWritten = 0;
	while (I2C_ERR_RESTART == status || I2C_ERR_ARB_LOST == status)
	{
		/*
		 * First cycle: master transmitter mode
		 */

		if (I2C_ERR_RESTART == status && iter++ >= MAX_RESTART_ITER_COUNT)
			return 0;

		// send start condition
		status = i2c_start();
		if (I2C_ERR_RESTART == status || I2C_ERR_ARB_LOST == status)
			continue;
		else if (I2C_ERR_QUIT_WITHOUT_STOP == status)
			return 0;

		// send SLA+W
		if (I2C_OK == status)
		{
			status = i2c_sla_w(sla);
			if (I2C_ERR_RESTART == status || I2C_ERR_ARB_LOST == status)
				continue;
			//else if (I2C_ERR_QUIT_WITH_STOP == status) ;
		}

		// first 8 bits of address
		if (I2C_OK == status)
		{
			status = i2c_address((eeaddr >> 8) & 0x00FF);
			if (I2C_ERR_ARB_LOST == status)
				continue;
			//else if (I2C_QUIT == status || I2C_ERR_QUIT_WITH_STOP == status) ;
		}

		// second 8 bits of address
		if (I2C_OK == status)
		{
			status = i2c_address(eeaddr & 0x00FF);
			if (I2C_ERR_ARB_LOST == status)
				continue;
			//else if (I2C_QUIT == status || I2C_ERR_QUIT_WITH_STOP == status) ;
		}

		// transmit bytes
		if (I2C_OK == status)
		{
			status = i2c_master_write_bytes(len, buf, byteLenWritten);
			if (I2C_ERR_ARB_LOST == status)
				continue;
			//else if (I2C_ERR_QUIT_WITH_STOP == status) ;
		}

		// send stop condition
		if (I2C_OK == status || I2C_QUIT == status || I2C_ERR_QUIT_WITH_STOP == status)
		{
			i2c_stop();
			return I2C_ERR_QUIT_WITH_STOP == status ? 0 : 1;
		}
	}

	return 1;
}

int ee24Cxxx_write_bytes(const uint16_t eeaddr, const uint16_t bufLen, const uint8_t *buf, uint16_t *byteLenWritten)
{
	uint16_t len = 0;

	uint16_t eeaddr2 = eeaddr;
	uint16_t bufLen2 = bufLen;
	do
	{
		len = 0;
		const int ret = ee24Cxx_write_a_page(eeaddr2, bufLen2, buf, &len);
		if (0 == ret) return 0;

		eeaddr2 += len;
		bufLen2 -= len;
		buf += len;
		*byteLenWritten += len;
	} while (bufLen2 > 0);

	return 1;
}

int ee24Cxxx_read_a_byte(const uint16_t eeaddr, uint8_t *byte)
{
	// patch hardwired input pins of EEPROM address into SLA
	const uint8_t sla = TWI_SLA_24CXXX;

	uint8_t iter = 0;
	I2C_STATUS status = I2C_ERR_RESTART;

	while (I2C_ERR_RESTART == status || I2C_ERR_ARB_LOST == status)
	{
		/*
		 * First cycle: master transmitter mode
		 */

		if (I2C_ERR_RESTART == status && iter++ >= MAX_RESTART_ITER_COUNT)
			return 0;

		// send start condition
		status = i2c_start();
		if (I2C_ERR_RESTART == status || I2C_ERR_ARB_LOST == status)
			continue;
		else if (I2C_ERR_QUIT_WITHOUT_STOP == status)
			return 0;

		// send SLA+W
		if (I2C_OK == status)
		{
			status = i2c_sla_w(sla);
			if (I2C_ERR_RESTART == status || I2C_ERR_ARB_LOST == status)
				continue;
			//else if (I2C_ERR_QUIT_WITH_STOP == status) ;
		}

		// first 8 bits of address
		if (I2C_OK == status)
		{
			status = i2c_address((eeaddr >> 8) & 0x00FF);
			if (I2C_ERR_ARB_LOST == status)
				continue;
			//else if (I2C_QUIT == status || I2C_ERR_QUIT_WITH_STOP == status) ;
		}

		// second 8 bits of address
		if (I2C_OK == status)
		{
			status = i2c_address(eeaddr & 0x00FF);
			if (I2C_ERR_ARB_LOST == status)
				continue;
			//else if (I2C_QUIT == status || I2C_ERR_QUIT_WITH_STOP == status) ;
		}

		/*
		 * Next cycle(s): master receiver mode
		 */

		// send (rep.) start condition
		if (I2C_OK == status)
		{
			status = i2c_repeated_start();
			if (I2C_ERR_RESTART == status || I2C_ERR_ARB_LOST == status)
				continue;
			//else if (I2C_ERR_QUIT_WITH_STOP == status) ;
		}

		// send SLA+R
		if (I2C_OK == status)
		{
			status = i2c_sla_r(sla);
			if (I2C_ERR_ARB_LOST == status)
				continue;
			//else if (I2C_QUIT == status || I2C_ERR_QUIT_WITH_STOP == status) ;
		}

		// receive a byte
		if (I2C_OK == status)
		{
			status = i2c_master_read_a_byte(byte);
			if (I2C_ERR_ARB_LOST == status)
				continue;
			//else if (I2C_ERR_QUIT_WITH_STOP == status) ;
		}

		// send stop condition
		if (I2C_OK == status || I2C_QUIT == status || I2C_ERR_QUIT_WITH_STOP == status)
		{
			i2c_stop();
			return I2C_ERR_QUIT_WITH_STOP == status ? 0 : 1;
		}
	}

	return 1;
}

int ee24Cxxx_read_bytes(const uint16_t eeaddr, const uint16_t bufLen, uint8_t *buf, uint16_t *byteLenRead)
{
	// patch hardwired input pins of EEPROM address into SLA
	const uint8_t sla = TWI_SLA_24CXXX;

	uint8_t iter = 0;
	I2C_STATUS status = I2C_ERR_RESTART;

	//*byteLenRead = 0;
	while (I2C_ERR_RESTART == status || I2C_ERR_ARB_LOST == status)
	{
		/*
		 * First cycle: master transmitter mode
		 */

		if (I2C_ERR_RESTART == status && iter++ >= MAX_RESTART_ITER_COUNT)
			return 0;

		// send start condition
		status = i2c_start();
		if (I2C_ERR_RESTART == status || I2C_ERR_ARB_LOST == status)
			continue;
		else if (I2C_ERR_QUIT_WITHOUT_STOP == status)
			return 0;

		// send SLA+W
		if (I2C_OK == status)
		{
			status = i2c_sla_w(sla);
			if (I2C_ERR_RESTART == status || I2C_ERR_ARB_LOST == status)
				continue;
			//else if (I2C_ERR_QUIT_WITH_STOP == status) ;
		}

		// first 8 bits of address
		if (I2C_OK == status)
		{
			status = i2c_address((eeaddr >> 8) & 0x00FF);
			if (I2C_ERR_ARB_LOST == status)
				continue;
			//else if (I2C_QUIT == status || I2C_ERR_QUIT_WITH_STOP == status) ;
		}

		// second 8 bits of address
		if (I2C_OK == status)
		{
			status = i2c_address(eeaddr & 0x00FF);
			if (I2C_ERR_ARB_LOST == status)
				continue;
			//else if (I2C_QUIT == status || I2C_ERR_QUIT_WITH_STOP == status) ;
		}

		/*
		 * Next cycle(s): master receiver mode
		 */

		// send (rep.) start condition
		if (I2C_OK == status)
		{
			status = i2c_repeated_start();
			if (I2C_ERR_RESTART == status || I2C_ERR_ARB_LOST == status)
				continue;
			//else if (I2C_ERR_QUIT_WITH_STOP == status) ;
		}

		// send SLA+R
		if (I2C_OK == status)
		{
			status = i2c_sla_r(sla);
			if (I2C_ERR_ARB_LOST == status)
				continue;
			//else if (I2C_QUIT == status || I2C_ERR_QUIT_WITH_STOP == status) ;
		}

		// receive bytes
		if (I2C_OK == status)
		{
			status = i2c_master_read_bytes(bufLen, buf, byteLenRead);
			if (I2C_ERR_ARB_LOST == status)
				continue;
			//else if (I2C_ERR_QUIT_WITH_STOP == status) ;
		}

		// send stop condition
		if (I2C_OK == status || I2C_QUIT == status || I2C_ERR_QUIT_WITH_STOP == status)
		{
			i2c_stop();
			return I2C_ERR_QUIT_WITH_STOP == status ? 0 : 1;
		}
	}

	return 1;
}
