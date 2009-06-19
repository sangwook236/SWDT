#include <avr/eeprom.h>
#include <avr/interrupt.h>
#include <avr/sleep.h>
#include <util/delay.h>


void initSystem();

/*
 * API for built-in EEPROM
 */
/*
eeprom_is_ready();
eeprom_busy_wait();

uint8_t eeprom_read_byte(const uint8_t *__p);
uint16_t eeprom_read_word(const uint16_t *__p);
uint32_t eeprom_read_dword(const uint32_t *__p);
void eeprom_read_block(void *__dst, const void *__src, size_t __n);

void eeprom_write_byte(uint8_t *__p, uint8_t __value);
void eeprom_write_word(uint16_t *__p, uint16_t __value);
void eeprom_write_dword(uint32_t *__p, uint32_t __value);
void eeprom_write_block(const void *__src, void *__dst, size_t __n);
*/

int main(void)
{
	void initPio();

	cli();
	initSystem();
	initPio();
	sei();

	PORTA = 0xFF;
	_delay_ms(50);
	PORTA = 0x00;

	//const int EEPROM_MAX_SIZE = 4000;  // 4 Kbytes
	const uint16_t eeaddr_start = 0x0000;

	const int len = 16;
	uint8_t buf[len];
	for (int i = 0; i < len; ++i)
	{
		//buf[i] = i + 1;
		//buf[i] = len - i;
		buf[i] = (i + 1) * 2;
	}

	uint16_t eeaddr = eeaddr_start;
	for (int i = 0; i < len; ++i, ++eeaddr)
	{
		//eeprom_busy_wait();  // don't need
		eeprom_write_byte((uint8_t *)eeaddr, buf[i]);  // caution !!! address
		//eeprom_write_word((uint16_t *)eeaddr, buf[i]);  // caution !!! address
		//eeprom_write_word((uint32_t *)eeaddr, buf[i]);  // caution !!! address

		PORTA = buf[i];
		_delay_ms(50);
	}

	PORTA = 0xFF;
	_delay_ms(50);
	PORTA = 0x00;

	eeaddr = eeaddr_start;
	for (int i = 0; i < len; ++i, ++eeaddr)
	{
		//eeprom_busy_wait();  // don't need
		const uint8_t val = eeprom_read_byte((uint8_t *)eeaddr);  // caution !!! address
		//const uint16_t val = eeprom_read_byte((uint16_t *)eeaddr);  // caution !!! address
		//const uint32_t val = eeprom_read_byte((uint32_t *)eeaddr);  // caution !!! address

		PORTA = val;
		_delay_ms(500);
	}

	PORTA = 0xFF;
	_delay_ms(50);
	PORTA = 0x00;

	for (;;)
		sleep_mode();

	return 0;
}

void initSystem()
{
	// Analog Comparator
	ACSR &= ~(_BV(ACIE));  // analog comparator interrupt disable
	ACSR |= _BV(ACD);  // analog comparator disable
}
