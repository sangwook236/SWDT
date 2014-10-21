#include <stm32f10x.h>
#include <stdio.h>


#if defined(__cplusplus)
extern "C" {
#endif

// For serial wire output (SWO).
// [ref] http://controlsoft.nmmu.ac.za/STM32F3-Discovery-Board/Example-programs/Serial-Wire-Viewer-(SWV).
/*
int fputc(int ch, FILE *f)
{
    // for USART.
    //--S [] 2014/10/20 : Sang-Wook Lee
    //UART4->TDR = (ch & (uint16_t)0x01FF);
    //while ((UART4->ISR & USART_FLAG_TXE) == (uint16_t) RESET);
    UART4->DR = (ch & (uint16_t)0x01FF);
    while ((UART4->SR & USART_FLAG_TXE) == (uint16_t) RESET);
    //--E [] 2014/10/20 : Sang-Wook Lee

    // for SWV.
    return ITM_SendChar(ch);
}
*/

#if defined(__cplusplus)
}
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
}

}  // namespace local
}  // unnamed namespace

namespace my_swd {

// [ref] _write_r() in ../Src/syscalls.c.
#define ITM_STIM_U32 (*(volatile unsigned int *)0xE0000000)    // Stimulus Port Register word acces.
#define ITM_STIM_U8  (*(volatile         char *)0xE0000000)    // Stimulus Port Register byte acces.
#define ITM_TER      (*(volatile unsigned int *)0xE0000E00)    // ITM Trace Enable Register.
#define ITM_TCR      (*(volatile unsigned int *)0xE0000E80)    // ITM Trace Control Register.
#define DHCSR        (*(volatile unsigned int *)0xE000EDF0)    // Debug Halting Control Status Register.
#define DEMCR        (*(volatile unsigned int *)0xE000EDFC)    // Debug Exception Monitor Control Register.
// [ref] http://community.arm.com/thread/2519.
#define ITM_LOCK_ACCESS_REGISTER  (*(volatile unsigned int *)0xE0000FB0)

void check_swd_1()
{
    // Initialize serial wire debugging (SWD).
    {
        DHCSR |= 0x00000001;  // Enable debug.
        DEMCR |= (1UL << 24);  // Enable TRCENA.

        // The ITM has a lock register that you need to unlock before programming the ITM.
        //ITM_LOCK_ACCESS_REGISTER = 0xC5ACCE55;

        ITM_TER |= 0x00000001;  // Enable ITM Port #0.
        ITM_TCR |= 0x00000001;  // Enable ITM.
    }

    while (true)
    {
        // [ref] _write_r() in ../Src/syscalls.c.
        if ((DHCSR & 0x00000001) == 1)
        {
            //
            GPIO_WriteBit(GPIOA, GPIO_Pin_0, Bit_SET);
            my_stm32::delay_ms(200);

            //
            GPIO_WriteBit(GPIOA, GPIO_Pin_0, Bit_RESET);
            my_stm32::delay_ms(200);
        }
        if (((DEMCR >> 24) & 1UL) == 1)
        {
            //
            GPIO_WriteBit(GPIOA, GPIO_Pin_1, Bit_SET);
            my_stm32::delay_ms(200);

            //
            GPIO_WriteBit(GPIOA, GPIO_Pin_1, Bit_RESET);
            my_stm32::delay_ms(200);
        }
        if ((ITM_TCR & 0x00000001) == 1)
        {
            //
            GPIO_WriteBit(GPIOA, GPIO_Pin_2, Bit_SET);
            my_stm32::delay_ms(200);

            //
            GPIO_WriteBit(GPIOA, GPIO_Pin_2, Bit_RESET);
            my_stm32::delay_ms(200);
        }
        if ((ITM_TER & 0x00000001) == 1)
        {
             //
            GPIO_WriteBit(GPIOA, GPIO_Pin_3, Bit_SET);
            my_stm32::delay_ms(100);

            //
            GPIO_WriteBit(GPIOA, GPIO_Pin_3, Bit_RESET);
            my_stm32::delay_ms(100);
        }

        GPIO_WriteBit(GPIOD, GPIO_Pin_2, Bit_SET);
        my_stm32::delay_ms(200);
        GPIO_WriteBit(GPIOD, GPIO_Pin_2, Bit_RESET);
        my_stm32::delay_ms(200);
    }
}

void check_swd_2()
{
    // Initialize serial wire debugging (SWD).
    {
        //CoreDebug->DHCSR |= 0x00000001;  // Enable debug.
        DHCSR |= 0x00000001;  // Enable debug.
        //CoreDebug->DEMCR |= (1UL << 24);  // Enable TRCENA.
        DEMCR |= (1UL << 24);  // Enable TRCENA.

        // The ITM has a lock register that you need to unlock before programming the ITM.
        //ITM->LAR = 0xC5ACCE55;

        ITM->TER |= (1UL << 0);  // ITM Trace Enable Register. Enable ITM Port #0.
        ITM->TCR |= ITM_TCR_ITMENA_Msk;  // ITM Trace Control Register. Enable ITM.
    }

    while (true)
    {
        // [ref] ITM_SendChar() in ../CMSIS/core_cm3.h.
        if ((ITM->TCR & ITM_TCR_ITMENA_Msk) && (ITM->TER & (1UL << 0)))  // ITM & ITM Port #0 enabled.
        {
            //
            GPIO_WriteBit(GPIOA, GPIO_Pin_0 | GPIO_Pin_1, Bit_SET);
            my_stm32::delay_ms(200);

            //
            GPIO_WriteBit(GPIOA, GPIO_Pin_0 | GPIO_Pin_1, Bit_RESET);
            my_stm32::delay_ms(200);
        }
        else if (ITM->TCR & ITM_TCR_ITMENA_Msk)  // ITM only enabled.
        {
            //
            GPIO_WriteBit(GPIOA, GPIO_Pin_0, Bit_SET);
            my_stm32::delay_ms(200);

            //
            GPIO_WriteBit(GPIOA, GPIO_Pin_0, Bit_RESET);
            my_stm32::delay_ms(200);
        }
        else if (ITM->TER & (1UL << 0))  // ITM Port #0 only enabled.
        {
            //
            GPIO_WriteBit(GPIOA, GPIO_Pin_1, Bit_SET);
            my_stm32::delay_ms(200);

            //
            GPIO_WriteBit(GPIOA, GPIO_Pin_1, Bit_RESET);
            my_stm32::delay_ms(200);
        }
        else
        {
            //
            GPIO_WriteBit(GPIOA, GPIO_Pin_1, Bit_SET);
            my_stm32::delay_ms(100);

            //
            GPIO_WriteBit(GPIOA, GPIO_Pin_1, Bit_RESET);
            my_stm32::delay_ms(100);

            //
            GPIO_WriteBit(GPIOA, GPIO_Pin_0, Bit_SET);
            my_stm32::delay_ms(100);

            //
            GPIO_WriteBit(GPIOA, GPIO_Pin_0, Bit_RESET);
            my_stm32::delay_ms(100);
       }

        GPIO_WriteBit(GPIOD, GPIO_Pin_2, Bit_SET);
        my_stm32::delay_ms(200);
        GPIO_WriteBit(GPIOD, GPIO_Pin_2, Bit_RESET);
        my_stm32::delay_ms(200);
    }
}

}  // namespace my_swd

// Serial wire debugging (SWD).
int swd_main(int argc, char *argv[])
{
    // Disable interrupt.
    local::system_init();
    // Enable interrupt.

    // For checking SWD.
    //my_swd::check_swd_1();
    //my_swd::check_swd_2();

    int i = 0, j = 0;
    while (true)
    {
        // Serial Wire Output (SWO) through ST-LINK/V2.
        // The Printf via SWO Viewer displays the printf data sent from the target through SWO.
        printf("\fprintf(%d, %d) test through SWO\n", ++i, ++j);

        my_stm32::delay_ms(500);

        // FIXME [delete] >> for test.
        // for SWV.
        ITM_SendChar('a');
        ITM_SendChar('\n');

        my_stm32::delay_ms(500);

        //
        GPIO_WriteBit(GPIOD, GPIO_Pin_2, Bit_SET);
        my_stm32::delay_ms(200);
        GPIO_WriteBit(GPIOD, GPIO_Pin_2, Bit_RESET);
        my_stm32::delay_ms(200);
    }

	return 0;
}
