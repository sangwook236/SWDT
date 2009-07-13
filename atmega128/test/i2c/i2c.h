#if !defined(__SWL_AVR__I2C_H_)
#define __SWL_AVR__I2C_H_ 1

#include <stdint.h>


//-----------------------------------------------------------------------------
//

typedef enum tagI2C_STATUS
{
	I2C_OK = 0,
	I2C_QUIT,
	I2C_ERR_RESTART,
	I2C_ERR_ARB_LOST,
	I2C_ERR_QUIT_WITHOUT_STOP,  // quit without sending stop condition
	I2C_ERR_QUIT_WITH_STOP  // quit after sending stop condition
} I2C_STATUS;


I2C_STATUS i2c_start();
I2C_STATUS i2c_repeated_start();
void i2c_stop();

I2C_STATUS i2c_sla_w(const uint8_t sla);
I2C_STATUS i2c_sla_r(const uint8_t sla);
I2C_STATUS i2c_address(const uint8_t addr);

I2C_STATUS i2c_master_write_a_byte(const uint8_t byte);
// write data to slave: buf[0](MSB) to buf[n](LSB)
I2C_STATUS i2c_master_write_bytes(const uint16_t bufLen, const uint8_t *buf, uint16_t *byteLenWritten);
I2C_STATUS i2c_master_read_a_byte(uint8_t *byte);
// read data from slave: buf[0](MSB) to buf[n](LSB)
I2C_STATUS i2c_master_read_bytes(const uint16_t bufLen, uint8_t *buf, uint16_t *byteLenRead);


#endif  // __SWL_AVR__I2C_H_


