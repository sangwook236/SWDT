//#define __USE_COOCOX_CoX_LIBRARY 1

#if defined(__USE_COOCOX_CoX_LIBRARY)

#include <xhw_types.h>
#include <xhw_memmap.h>
#include <xsysctl.h>
#include <xgpio.h>
#include <stm32f10x.h>

#else

#include <stm32f10x.h>
#include <cstdio>

#endif


namespace my_stm32 {

void delay_ms(const uint16_t ms);
void delay_us(const uint16_t us);
void delay_ns(const uint16_t ns);

}  // namespace my_stm32

namespace {
namespace local {

void system_init()
{
#if defined(__USE_COOCOX_CoX_LIBRARY)
    // Initialize system clock.
    xSysCtlPeripheralClockSourceSet(12000000,  xSYSCTL_XTAL_12MHZ);

    // Set GPIO port A pin 0 & 1 output mode.
    xGPIODirModeSet(xGPIO_PORTA_BASE, xGPIO_PIN_0, xGPIO_DIR_MODE_OUT);
    xGPIODirModeSet(xGPIO_PORTA_BASE, xGPIO_PIN_1, xGPIO_DIR_MODE_OUT);
    //xGPIOSDirModeSet(PA0, xGPIO_DIR_MODE_OUT);
    //xGPIOSDirModeSet(PA1, xGPIO_DIR_MODE_OUT);
    //XPinTypeGPIOOutput(xGPIO_PORTA_BASE, xGPIO_PIN_0 | xGPIO_PIN_1);

    // Set GPIO port B pin 8 & 9 output mode.
    //xGPIODirModeSet(xGPIO_PORTB_BASE, xGPIO_PIN_8, xGPIO_DIR_MODE_OUT);
    //xGPIODirModeSet(xGPIO_PORTB_BASE, xGPIO_PIN_9, xGPIO_DIR_MODE_OUT);
    //xGPIOSDirModeSet(PB8, xGPIO_DIR_MODE_OUT);
    //xGPIOSDirModeSet(PB9, xGPIO_DIR_MODE_OUT);
    //XPinTypeGPIOOutput(xGPIO_PORTB_BASE, xGPIO_PIN_8 | xGPIO_PIN_9);

    // Set GPIO port D pin 2 output mode.
    //xGPIODirModeSet(xGPIO_PORTD_BASE, xGPIO_PIN_2, xGPIO_DIR_MODE_OUT);
#else
    // Initialize STM32 board.
    GPIO_InitTypeDef GPIO_InitStructure;

    {
        // Initialize LED which connected to PA0 & 1, Enable the Clock.
        RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA, ENABLE);

        // Configure the GPIO_LED pin.
        GPIO_InitStructure.GPIO_Pin = GPIO_Pin_0 | GPIO_Pin_1;
        GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
        GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
        GPIO_Init(GPIOA, &GPIO_InitStructure);
    }

    {
        // Initialize LED which connected to PB8 & 9, Enable the Clock.
        RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOB, ENABLE);

        // Configure the GPIO_LED pin.
        GPIO_InitStructure.GPIO_Pin = GPIO_Pin_8 | GPIO_Pin_9;
        GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
        GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
        GPIO_Init(GPIOB, &GPIO_InitStructure);
    }

    {
        // Initialize LED which connected to PD2, Enable the Clock.
        RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOD, ENABLE);

        // Configure the GPIO_LED pin.
        GPIO_InitStructure.GPIO_Pin = GPIO_Pin_2;
        GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
        GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
        GPIO_Init(GPIOD, &GPIO_InitStructure);
    }
#endif
}

void simple_blink_example()
{
#if defined(__USE_COOCOX_CoX_LIBRARY)
    // Error : not working.

    // [ref] http://www.coocox.org/forum/topic.php?id=2125.
    while (true)
    {
        //for (unsigned long i = 0; i < 0x1FFFF; ++i)
        {
            // Output high level.
            xGPIOPinWrite(xGPIO_PORTA_BASE, xGPIO_PIN_0 | xGPIO_PIN_1, 0xFFFF);
            //xGPIOSPinWrite(PA0, 1);
            //xGPIOSPinWrite(PA1, 1);

            //xGPIOPinWrite(xGPIO_PORTB_BASE, xGPIO_PIN_8 | xGPIO_PIN_9, 0xFFFF);
            //xGPIOSPinWrite(PB8, 1);
            //xGPIOSPinWrite(PB9, 1);

            //xGPIOPinWrite(xGPIO_PORTD_BASE, xGPIO_PIN_2, 1);

            my_stm32::delay_ms(200);
        }

        //for (unsigned long i = 0; i < 0x1FFFF; ++i)
        {
            // Output low level.
            xGPIOPinWrite(xGPIO_PORTA_BASE, xGPIO_PIN_0 | xGPIO_PIN_1, 0x0000);
            //xGPIOSPinWrite(PA0, 0);
            //xGPIOSPinWrite(PA1, 0);

            //xGPIOPinWrite(xGPIO_PORTB_BASE, xGPIO_PIN_8 | xGPIO_PIN_9, 0x0000);
            //xGPIOSPinWrite(PB8, 0);
            //xGPIOSPinWrite(PB9, 0);

            //xGPIOPinWrite(xGPIO_PORTD_BASE, xGPIO_PIN_2, 0);

            my_stm32::delay_ms(200);
        }
    }
#else
    // [ref] http://www.coocox.org/forum/topic.php?id=1045.
    while (true)
    {
        //
        GPIO_WriteBit(GPIOA, GPIO_Pin_0 | GPIO_Pin_1, Bit_SET);
        //GPIO_WriteBit(GPIOB, GPIO_Pin_8 | GPIO_Pin_9, Bit_SET);
        GPIO_WriteBit(GPIOD, GPIO_Pin_2, Bit_SET);
        my_stm32::delay_ms(200);

        //
        GPIO_WriteBit(GPIOA, GPIO_Pin_0 | GPIO_Pin_1, Bit_RESET);
        //GPIO_WriteBit(GPIOB, GPIO_Pin_8 | GPIO_Pin_9, Bit_RESET);
        GPIO_WriteBit(GPIOD, GPIO_Pin_2, Bit_RESET);
        my_stm32::delay_ms(200);

        // Serial Wire Output (SWO) through ST-LINK/V2.
        // The Printf via SWO Viewer displays the printf data sent from the target through SWO.
        printf("printf() test through SWO\n");
    }
#endif
}

}  // namespace local
}  // unnamed namespace

namespace my_gpio {
}  // namespace my_gpio

int gpio_main(int argc, char *argv[])
{
    // Disable interrupt.
    local::system_init();
    // Enable interrupt.

	local::simple_blink_example();

	return 0;
}
