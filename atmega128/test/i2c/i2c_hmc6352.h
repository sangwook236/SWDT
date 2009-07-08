#if !defined(__SWL_AVR__I2C_HMC6352_H_)
#define __SWL_AVR__I2C_HMC6352_H_ 1


#include <stdint.h>


//-----------------------------------------------------------------------------
// for HMC6352: magnetometer

// HMC6352 interface commands

extern const char HMC6352_CMD_S;  // enter sleep mode (Sleep)
extern const char HMC6352_CMD_W;  // exit sleep mode (Wakeup)
extern const char HMC6352_CMD_O;  // update bridge offset (S/R Now)
extern const char HMC6352_CMD_C;  // enter user calibration mode
extern const char HMC6352_CMD_E;  // exit user calibration mode
extern const char HMC6352_CMD_L;  // save OP mode to EEPROM

int hmc6352_get_data(uint8_t *data);  // read 2-byte data. use command 'A'. for stand-by mode.
int hmc6352_read_one_byte_data(uint8_t *datum);  // read 1-byte data. do not use any command. for query and continuous modes.
int hmc6352_read_two_byte_data(uint8_t *data);  // read 2-byte data. do not use any command. for query and continuous modes.

int hmc6352_write_to_eeprom(const uint8_t addr, const uint8_t byte);  // write a byte to EEPROM.
int hmc6352_read_from_eeprom(const uint8_t addr, uint8_t *byte);  // read a byte from EEPROM.
int hmc6352_write_to_ram(const uint8_t addr, const uint8_t byte);  // write a byte to RAM register.
int hmc6352_read_from_ram(const uint8_t addr, uint8_t *byte);  // read a byte from RAM register.

int hmc6352_write_command(const char cmd);  // write a single command.


#endif  // __SWL_AVR__I2C_HMC6352_H_
