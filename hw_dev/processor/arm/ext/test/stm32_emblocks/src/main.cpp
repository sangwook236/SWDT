#include <stm32f10x.h>
#include <cstdlib>


namespace my_stm32 {

// [ref]
//  https://github.com/mikeferguson/stm32/blob/master/libraries/libcpp/inc/delay.hpp
//  https://github.com/mikeferguson/stm32/blob/master/libraries/libcpp/src/delay.cpp

// 72MHz = each clock cycle is 14ns.
// Loop is always 6 clock cycles. (???)
// These can get clock stretched if we have interrupts in the background.

void delay_ms(const uint16_t ms)
{
    uint32_t i = ms * 12000;  // 1ms/((1/72MHz)*6)= 12000.
    while (--i > 0)
    {
        asm("nop");
    }
}

void delay_us(const uint16_t us)
{
    uint32_t i = us * 12;  // 1us/((1/72MHz)*6)= 12.
    while (--i > 0)
    {
        asm("nop");
    }
}

void delay_ns(const uint16_t ns)
{
    uint32_t i = ns * 12 / 1000;  // 1ns/((1/72MHz)*6)= 0.012.
    while (--i > 0)
    {
        asm("nop");
    }
}

}  // namespace my_stm32

// NOTICE [caution] >> Important !!!
//  Define USE_FULL_ASSERT in order to use full function of assert_param() in ../inc/stm32f10x_conf.h.
//      #define USE_FULL_ASSERT 1

int main(int argc, char *argv[])
{
    int swd_main(int argc, char *argv[]);
    int gpio_main(int argc, char *argv[]);

 	int retval = EXIT_SUCCESS;
	{
        retval = swd_main(argc, argv);
        //retval = gpio_main(argc, argv);
	}

	return retval;
}
