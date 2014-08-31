#include "global.h"
#include "spi.h"
#include "spieeprom.h"
#include <avr/sleep.h>
#include <avr/interrupt.h>
#include <util/delay.h>


void initSystem();
void delay_msec(uint16_t msec);
void selectSpiSlave();
void deselectSpiSlave();
void waitInProgress();
void enableWriteMode();
void my_spieepromWriteStatus(u08 status);

int main(void)
{
	void initPio();
	void initSpi();
	void displayDigitsOnCathodeCommonFnd(uint16_t fourDigits);

	cli();
	initSystem();
	initPio();
	//initSpi();
	spiInit();
	spieepromInit();
	sei();

	PORTA = 0xFF;
	delay_msec(100);
	PORTA = 0x00;

	// WP(write protection) enable(low) or disable(high)
	PORTB |= _BV(PB4);  // high
	delay_msec(500);
/*
	// test status register read/write
	{
		// 1. select a slave
		selectSpiSlave();
		// 2. do operation
		my_spieepromWriteStatus(0x8C);
		// 3. deselect the slave
		deselectSpiSlave();
	}
	delay_msec(500);
*/
	uint8_t status = 0x00;
	{
		// 1. select a slave
		selectSpiSlave();
		// 2. do operation
		status = spieepromReadStatus();
		// 3. deselect the slave
		deselectSpiSlave();
	}
	waitInProgress();

	PORTA = status;
	delay_msec(1000);

	//
	u32 memAddr = 0x0100;

	//
	u08 writtenData = 0xAA;

	PORTA = writtenData;
	delay_msec(500);

	//
	enableWriteMode();
	delay_msec(1000);
	{
		// 1. select a slave
		selectSpiSlave();
		// 2. do operation
		spieepromWriteByte(memAddr, writtenData);
		// 3. deselect the slave
		deselectSpiSlave();
	}
	waitInProgress();

	//
	PORTA = 0xF0;
	delay_msec(500);

	//
	u08 readData = 0xFF;
	{
		// 1. select a slave
		selectSpiSlave();
		// 2. do operation
		readData = spieepromReadByte(memAddr);
		// 3. deselect the slave
		deselectSpiSlave();
	}
	waitInProgress();

	//
	PORTA = readData;
	delay_msec(500);

	for ( ; ; )
		sleep_mode();

	return 0;
}

void initSystem()
{
	// Analog Comparator
	ACSR &= ~(_BV(ACIE));  // analog comparator interrupt disable
	ACSR |= _BV(ACD);  // analog comparator disable
}

void delay_msec(uint16_t msec)
{
	uint16_t loop_count = msec / 10;
	uint16_t remainder = msec % 10;
	// a maximal possible delay time is (262.14 / Fosc in MHz) ms
	// if Fosc = 16 MHz, a maximal possible delay time = 16.38375 ms
	//	500 ms -> 10 ms * 50 count
	for (int k = 0; k < loop_count; ++k)
		_delay_ms(10);
	_delay_ms(remainder);
}

void selectSpiSlave()
{
	PORTB &= ~(_BV(PB0));  // low
	//delay_msec(100);
}

void deselectSpiSlave()
{
	PORTB |= _BV(PB0);  // high
	//delay_msec(100);
}

void my_spieepromWriteStatus(u08 status)
{
	// wait for any previous write to complete
	while(spieepromReadStatus() & SPIEEPROM_STATUS_WIP);

//	cbi(PORTB,0);
	// send command
	spiTransferByte(SPIEEPROM_CMD_WRSR);
	//spiSendByte(SPIEEPROM_CMD_WRSR);
	// set status value
	spiTransferByte(status);
	//spiSendByte(status);
//	sbi(PORTB,0);
}

void waitInProgress()
{
	uint8_t status = 0x00;
	do
	{
		// 1. select a slave
		selectSpiSlave();
		// 2. do operation
		status = spieepromReadStatus();
		// 3. deselect the slave
		deselectSpiSlave();
	} while ((status & 0x01) == 0x01);
}

void enableWriteMode()
{
	{
		// 1. select a slave
		selectSpiSlave();
		// 2. do operation
		spieepromWriteEnable();
		// 3. deselect the slave
		deselectSpiSlave();
	}
}
