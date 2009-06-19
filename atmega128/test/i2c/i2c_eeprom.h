#if !defined(__SWL_AVR__I2C_EEPROM_H_)
#define __SWL_AVR__I2C_EEPROM_H_ 1

#include <stdint.h>


//-----------------------------------------------------------------------------
// for 24C01/24C02/24C04/24C08/24C16

int ee24Cxx_write_a_byte(const uint16_t eeaddr, const uint8_t byte);
int ee24Cxx_write_bytes(const uint16_t eeaddr, const uint16_t bufLen, const uint8_t *buf, uint16_t *byteLenWritten);

int ee24Cxx_read_a_byte(const uint16_t eeaddr, uint8_t *byte);
int ee24Cxx_read_bytes(const uint16_t eeaddr, const uint16_t bufLen, uint8_t *buf, uint16_t *byteLenRead);

//-----------------------------------------------------------------------------
// for 24C128/24C256

int ee24Cxxx_write_a_byte(const uint16_t eeaddr, const uint8_t byte);
int ee24Cxxx_write_bytes(const uint16_t eeaddr, const uint16_t bufLen, const uint8_t *buf, uint16_t *byteLenWritten);

int ee24Cxxx_read_a_byte(const uint16_t eeaddr, uint8_t *byte);
int ee24Cxxx_read_bytes(const uint16_t eeaddr, const uint16_t bufLen, uint8_t *buf, uint16_t *byteLenRead);


#endif  // __SWL_AVR__I2C_EEPROM_H_
