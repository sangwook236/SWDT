#include "i2c_hmc6352.h"
#include "i2c.h"
#include <util/twi.h>
#include <util/delay.h>


//-----------------------------------------------------------------------------
//

// HMC6352 interface commands.

const char HMC6352_CMD_w = 'w';  // Write to EEPROM.
const char HMC6352_CMD_r = 'r';  // Read from EEPROM.
const char HMC6352_CMD_G = 'G';  // Write to RAM register.
const char HMC6352_CMD_g = 'g';  // Read from RAM register.
const char HMC6352_CMD_S = 'S';  // Enter sleep mode (Sleep).
const char HMC6352_CMD_W = 'W';  // Exit sleep mode (Wakeup).
const char HMC6352_CMD_O = 'O';  // Update bridge offset (S/R Now).
const char HMC6352_CMD_C = 'C';  // Enter user calibration mode.
const char HMC6352_CMD_E = 'E';  // Exit user calibration mode.
const char HMC6352_CMD_L = 'L';  // Save OP mode to EEPROM.
const char HMC6352_CMD_A = 'A';  // Get data. compensate & calculate new heading.

// EEPROM addresses.

const uint8_t HMC6352_EEPROM__I2C_SLAVE_ADDR = 0x00;
const uint8_t HMC6352_EEPROM__X_OFFSET_MSB = 0x01;
const uint8_t HMC6352_EEPROM__X_OFFSET_LSB = 0x02;
const uint8_t HMC6352_EEPROM__Y_OFFSET_MSB = 0x03;
const uint8_t HMC6352_EEPROM__Y_OFFSET_LSB = 0x04;
const uint8_t HMC6352_EEPROM__TIME_DELAY = 0x05;
const uint8_t HMC6352_EEPROM__NUM_SUMMED_MEASUREMENT = 0x06;
const uint8_t HMC6352_EEPROM__SW_VER = 0x07;
const uint8_t HMC6352_EEPROM__OPERATIONAL_MODE = 0x08;

// RAM register addresses.

const uint8_t HMC6352_RAM__OPERATIONAL_MODE = 0x74;
const uint8_t HMC6352_RAM__OUTPUT_DATA_MODE = 0x4E;

//-----------------------------------------------------------------------------
//

int hmc6352_write_a_byte(const char cmd, const uint8_t addr, const uint8_t byte);
int hmc6352_read_a_byte(const char cmd, const uint8_t addr, uint8_t *byte);

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
 * TWI address for HMC6352.
 */
static const uint8_t TWI_SLA_HMC6352 = 0x42;


int hmc6352_get_data(uint8_t *data)
{
	// Patch HMC6352 address into SLA.
	const uint8_t sla = TWI_SLA_HMC6352;
	const char cmd = HMC6352_CMD_A;

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
			status = i2c_master_write_a_byte(cmd);
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
	const uint16_t dataLen = 2;
	uint16_t dataLenRead = 0;

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
			status = i2c_master_read_bytes(dataLen, data, &dataLenRead);
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

	return dataLen == dataLenRead ? 1 : 0;
}

int hmc6352_read_byte_data(uint8_t *datum)
{
	// Patch HMC6352 address into SLA.
	const uint8_t sla = TWI_SLA_HMC6352;

	/*
	 * First cycle: Master receiver mode.
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
			status = i2c_master_read_a_byte(datum);
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

int hmc6352_read_word_data(uint8_t *data)
{
	// Patch HMC6352 address into SLA.
	const uint8_t sla = TWI_SLA_HMC6352;
	const uint16_t dataLen = 2;
	uint16_t dataLenRead = 0;

	/*
	 * First cycle: Master receiver mode.
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
			status = i2c_master_read_bytes(dataLen, data, &dataLenRead);
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

	return dataLen == dataLenRead ? 1 : 0;
}

int hmc6352_write_to_eeprom(const uint8_t addr, const uint8_t byte)
{
	return hmc6352_write_a_byte(HMC6352_CMD_w, addr, byte);
}

int hmc6352_read_from_eeprom(const uint8_t addr, uint8_t *byte)
{
	return hmc6352_read_a_byte(HMC6352_CMD_r, addr, byte);
}

int hmc6352_write_to_ram(const uint8_t addr, const uint8_t byte)
{
	return hmc6352_write_a_byte(HMC6352_CMD_G, addr, byte);
}

int hmc6352_read_from_ram(const uint8_t addr, uint8_t *byte)
{
	return hmc6352_read_a_byte(HMC6352_CMD_g, addr, byte);
}

int hmc6352_write_command(const char cmd)
{
	// Patch DS1307 address into SLA.
	const uint8_t sla = TWI_SLA_HMC6352;

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

		// Transmit a command.
		if (I2C_OK == status)
		{
			status = i2c_master_write_a_byte(cmd);
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

//-----------------------------------------------------------------------------
//

int hmc6352_write_a_byte(const char cmd, const uint8_t addr, const uint8_t byte)
{
	// Patch DS1307 address into SLA.
	const uint8_t sla = TWI_SLA_HMC6352;

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

		// Command.
		if (I2C_OK == status)
		{
			status = i2c_master_write_a_byte(cmd);
			if (I2C_ERR_ARB_LOST == status)
				continue;
			//else if (I2C_QUIT == status || I2C_ERR_QUIT_WITH_STOP == status) ;
		}

		// 8-bit EEPROM or RAM address.
		if (I2C_OK == status)
		{
			status = i2c_master_write_a_byte(addr);
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

int hmc6352_read_a_byte(const char cmd, const uint8_t addr, uint8_t *byte)
{
	// Patch HMC6352 address into SLA.
	const uint8_t sla = TWI_SLA_HMC6352;

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

		// Command.
		if (I2C_OK == status)
		{
			status = i2c_master_write_a_byte(cmd);
			if (I2C_ERR_ARB_LOST == status)
				continue;
			//else if (I2C_QUIT == status || I2C_ERR_QUIT_WITH_STOP == status) ;
		}

		// 8-bit EEPROM or RAM address.
		if (I2C_OK == status)
		{
			status = i2c_master_write_a_byte(addr);
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

