#if !defined(__SWL_AVR__SPI_EEPROM_H_)
#define __SWL_AVR__SPI_EEPROM_H_ 1

#include <stdint.h>


//-----------------------------------------------------------------------------
// for 93C46/93C56/93C66: use 8-bit access

void ee93Cxx_init();
void ee93Cxx_chip_select();
void ee93Cxx_chip_deselect();

int ee93Cxx_set_write_enable();
int ee93Cxx_set_write_disable();

int ee93Cxx_write_a_byte(const uint16_t eeaddr, const uint8_t byte);
int ee93Cxx_read_a_byte(const uint16_t eeaddr, uint8_t *byte);

int ee93Cxx_erase(const uint16_t eeaddr);
int ee93Cxx_erase_all();
int ee93Cxx_write_all(const uint8_t byte);

//-----------------------------------------------------------------------------
// for 25128/25256

void ee25xxx_init();
void ee25xxx_chip_select();
void ee25xxx_chip_deselect();

int ee25xxx_set_write_enable();
int ee25xxx_set_write_disable();

int ee25xxx_write_status_register(const uint8_t byte);
int ee25xxx_read_status_register(uint8_t *byte);

int ee25xxx_is_ready();

int ee25xxx_write_a_byte(const uint16_t eeaddr, const uint8_t byte);
int ee25xxx_read_a_byte(const uint16_t eeaddr, uint8_t *byte);


#endif  // __SWL_AVR__SPI_EEPROM_H_
