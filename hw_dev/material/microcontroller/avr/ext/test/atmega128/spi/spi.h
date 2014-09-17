#if !defined(__SWL_AVR__SPI_H_)
#define __SWL_AVR__SPI_H_ 1

#include <stdint.h>


//#define __SWL_AVR__USE_SPI_INTERRUPT 1

#if defined(__SWL_AVR__USE_SPI_INTERRUPT)
#define spi_enable_interrupt() SPCR |= _BV(SPIE)
#define spi_disable_interrupt() SPCR &= ~(_BV(SPIE))
#else
#define spi_enable_interrupt()
#define spi_disable_interrupt()
#endif  // __SWL_AVR__USE_SPI_INTERRUPT


typedef enum tagSPI_STATUS
{
	SPI_OK = 0,
	SPI_QUIT,
} SPI_STATUS;


uint8_t spi_is_master();

#if defined(__SWL_AVR__USE_SPI_INTERRUPT)
uint8_t spi_is_transmit_busy();
#endif  // __SWL_AVR__USE_SPI_INTERRUPT

uint8_t spi_master_transmit_a_byte(const uint8_t byte);
void spi_master_transmit_bytes(const uint8_t *buf, const uint16_t lengthToBeWritten);
void spi_master_transmit_a_string(const uint8_t *str);
uint8_t spi_master_receive_a_byte();
void spi_master_receive_bytes(uint8_t *buf, const uint16_t lengthToBeRead);


#endif  // __SWL_AVR__SPI_H_


