#include <avr/io.h>
#include <avr/interrupt.h>
#include <util/delay.h>


namespace {
namespace local {

void system_init()
{
	// analog comparator.
	ACSR &= ~(_BV(ACIE));  // analog comparator interrupt disable.
	ACSR |= _BV(ACD);  // analog comparator disable.

	// I/O port.
#if 0
	// uses all pins on PortA for input.
   	//outp(0x00,DDRA);
	DDRA = 0x00;
	// it makes port input register(PINn) internally pulled-up state that port output register(PORTn) outputs 1(high).
	PORTA = 0xFF;
	// it makes port input register(PINn) high-impedance state that port output register(PORTn) outputs 0(low)
	// so that we can share the pin with other devices.
	//PORTA = 0x00;

	// uses all pins on PortD for output.
   	//outp(0xFF,DDRD);
	DDRD = 0xFF;
#else
	DDRA = 0xFF;  // uses all pins on PortA for output.
#endif
}

void pio_test_1()
{
  	uint8_t led = 1;  // init variable representing the LED state.

	while (1)
	{
		//outp(led, PORTA);  // invert the output since a zero means: LED on.
		PORTA = led;
		led <<= 1;  // move to next LED
		if (!led)  // overflow: start with Port C0 again.
		led = 1;

#if 0
		for (int i = 0; i < 30000; ++i)  // outer delay loop.
		for(int j = 0; j < 30000; ++j)  // inner delay loop.
		++k;  // just do something - could also be a NOP.
#else
		// a maximal possible delay time is (262.14 / Fosc in MHz) ms.
		// if Fosc = 16 MHz, a maximal possible delay time = 16.38375 ms.
		// 500 ms -> 10 ms * 50 count
		for (int i = 0; i < 50; ++i)
			_delay_ms(10);
#endif
	}
}

void pio_test_2()
{
	char *PINF_ptr = (char *)0x20;  // PINF => 0x20 (address in SRAM).
	//char *PINE_ptr = (char *)0x21;  // PINE => 0x21 (address in SRAM).
	char *DDRE_ptr = (char *)0x22;  // DDRE => 0x22 (address in SRAM).
	char *PORTE_ptr = (char *)0x23;  // PORTE => 0x23 (address in SRAM).

	int8_t led = 0;

	//DDRE = 0xFF;  // 1 for output, 0 for input.
	*DDRE_ptr = 0xFF;  // 1 for output, 0 for input.
	DDRF = 0x00;  // 1 for output, 0 for input.

	while (1)
	{
#if 0
		PORTE = 0xAA;
		_delay_ms(500);

		//PORTE = 0x55;
		*PORTE_ptr = 0xFF;
		_delay_ms(500);
#else
		//led = PINF;
		led = *PINF_ptr;

		//PORTE = led;
		*PORTE_ptr = led;
		_delay_ms(500);
#endif
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_gpio {
}  // namespace my_gpio

int gpio_main(int argc, char *argv[])
{
	cli();
	local::system_init();
	sei();

	//local::pio_test_1();
	local::pio_test_2();

	return 0;
}
