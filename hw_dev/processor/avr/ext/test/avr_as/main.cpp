#include <stdlib.h>


int main(void)
{
	int eeprom_main();
	int flash_main();
	int wdt_main();

	int pio_main();
	int interrupt_main();
	int pwm_main();
	int adc_main();

	int usart_main();
	int spi_main();
	int i2c_main();
	int zigbee_main();

	int seven_segment_main();
	int motor_main();

	// Important !!!
	// In order to make use of functions, do set "Properties -> Build Action" to "None -> Compile".

	int retval = EXIT_SUCCESS;
	{
		//retval = eeprom_main();
		//retval = flash_main();
		//retval = wdt_main();

		retval = pio_main();
		//retval = interrupt_main();
		//retval = pwm_main();
		//retval = adc_main();

		//retval = usart_main();
		//retval = spi_main();
		//retval = i2c_main();
		//retval = zigbee_main();

		//retval = seven_segment_main();
		//retval = motor_main();
	}

	return retval;
}