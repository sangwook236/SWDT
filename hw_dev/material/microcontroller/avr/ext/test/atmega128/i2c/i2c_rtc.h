#if !defined(__SWL_AVR__I2C_RTC_H_)
#define __SWL_AVR__I2C_RTC_H_ 1

#include <stdint.h>


//-----------------------------------------------------------------------------
// for DS1307

int ds1307_init();

int ds1307_set_date_time(const uint8_t year, const uint8_t month, const uint8_t date, const uint8_t day_of_week, const uint8_t hour, const uint8_t minute, const uint8_t second);
int ds1307_get_time(uint8_t *hour, uint8_t *minute, uint8_t *second);
int ds1307_get_date(uint8_t *year, uint8_t *month, uint8_t *date, uint8_t *day_of_week);


#endif  // __SWL_AVR__I2C_RTC_H_
