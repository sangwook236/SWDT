#if !defined(__SWL_AVR__SPI_ADIS16350_H_)
#define __SWL_AVR__SPI_ADIS16350_H_ 1


#include <stdint.h>


//-----------------------------------------------------------------------------
// for ADIS16350

#define ADIS16350_SUPPLY_OUT	0x02
#define ADIS16350_XGYRO_OUT		0x04
#define ADIS16350_YGYRO_OUT		0x06
#define ADIS16350_ZGYRO_OUT		0x08
#define ADIS16350_XACCL_OUT		0x0A
#define ADIS16350_YACCL_OUT		0x0C
#define ADIS16350_ZACCL_OUT		0x0E
#define ADIS16350_XTEMP_OUT		0x10
#define ADIS16350_YTEMP_OUT		0x12
#define ADIS16350_ZTEMP_OUT		0x14
#define ADIS16350_AUX_ADC		0x16

#define ADIS16350_COMMAND		0x3E
#define ADIS16350_SMPL_PRD		0x36
#define ADIS16350_SENS_AVG		0x38
#define ADIS16350_MSC_CTRL		0x34
#define ADIS16350_STATUS		0x3C

void adis16350_init();
void adis16350_chip_select();
void adis16350_chip_deselect();

int adis16350_write_a_register(const uint8_t addr, const uint16_t word);
int adis16350_read_a_register(const uint8_t addr, uint16_t *word);


#endif  // __SWL_AVR__SPI_ADIS16350_H_
