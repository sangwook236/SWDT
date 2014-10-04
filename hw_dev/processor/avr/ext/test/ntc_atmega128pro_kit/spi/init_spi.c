#include <avr/interrupt.h>
#include <avr/sfr_defs.h>


static volatile int8_t gv_isSpiMaster = 0;

void setSpiMaster(int8_t isSpiMaster)
{  gv_isSpiMaster = isSpiMaster;  }

int8_t isSpiMaster()
{  return gv_isSpiMaster;  }
/*
ISR(SPI_STC_vect)
{
	if (gv_isSpiMaster) {
		if ((SPCR & (uint8_t)_BV(MSTR)) != (uint8_t)_BV(MSTR)) {
			SPCR |= _BV(MSTR);  // master select
			return;
		}
	}
	else {
	}
}
*/
void initSpi()
{
	SPCR = 0x00;
	SPSR = 0x00;
	SPDR = 0;

	SPCR |= _BV(SPE);  // SPI enable
	SPCR |= _BV(DORD);  // 0 -> MSB to LSB. 1 -> LSB to MSB

	if (gv_isSpiMaster) {
		SPCR |= _BV(MSTR);  // master select
	}
	else {
		SPCR &= ~(_BV(MSTR));  // slave select
	}

	// SPI mode 2
	// clock polarity: falling edge. clock phase: leading edge
	SPCR |= _BV(CPOL);
	SPCR &= ~(_BV(CPHA));

	// clock rate: Fosc / 4
	SPSR &= ~(_BV(SPI2X));
	SPCR &= ~(_BV(SPR1));
	SPCR &= ~(_BV(SPR0));
}
