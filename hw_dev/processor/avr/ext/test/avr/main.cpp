#include <stdlib.h>


int main(int argc, char *argv[])
{
	int eeprom_main(int argc, char *argv[]);
	int flash_main(int argc, char *argv[]);
	int wdt_main(int argc, char *argv[]);

	int gpio_main(int argc, char *argv[]);
	int interrupt_main(int argc, char *argv[]);
	int pwm_main(int argc, char *argv[]);
	int adc_main(int argc, char *argv[]);

	int usart_main(int argc, char *argv[]);
	int spi_main(int argc, char *argv[]);
	int i2c_main(int argc, char *argv[]);
	int zigbee_main(int argc, char *argv[]);

	int seven_segment_main(int argc, char *argv[]);
	int motor_main(int argc, char *argv[]);

	// NOTICE [caution] >> Important !!!
	// In order to make use of functions, do set "Properties -> Build Action" to "None -> Compile".

	int retval = EXIT_SUCCESS;
	{
		//retval = eeprom_main(argc, argv);
		//retval = flash_main(argc, argv);
		//retval = wdt_main(argc, argv);

		retval = gpio_main(argc, argv);
		//retval = interrupt_main(argc, argv);
		//retval = pwm_main(argc, argv);
		//retval = adc_main(argc, argv);

		//retval = usart_main(argc, argv);
		//retval = spi_main(argc, argv);
		//retval = i2c_main(argc, argv);
		//retval = zigbee_main(argc, argv);

		//retval = seven_segment_main(argc, argv);
		//retval = motor_main(argc, argv);
	}

	return retval;
}
