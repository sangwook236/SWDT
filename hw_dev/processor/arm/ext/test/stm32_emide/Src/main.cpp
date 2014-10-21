#include <stm32f10x.h>
#include <cstdlib>
//#include <cerrno>


#if defined(__cplusplus)
extern "C" {
#endif

// These functions hav already been defined in ./syscalls.c.
/*
void _exit(int status)
{
    // do something if needed.
    while (true) ;
}

typedef int pid_t;

#undef errno
extern int errno;

int _kill(pid_t pid, int sig)
{
    // do something if needed.
    errno = EINVAL;
    return -1;
}

pid_t _getpid()
{
    // do something if needed.
    return -1;
}
*/

// SystemInit() has already been defined in ../CMSIS/system_stm32f10x.c.
/*
#if !defined(__NO_SYSTEM_INIT)

void SystemInit()  // use in ./Setup/startup.S.
{
    // do something if needed.
}

#endif  // __NO_SYSTEM_INIT
*/

#if defined(__cplusplus)
}
#endif

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
//  Define USE_FULL_ASSERT in order to use full function of assert_param() in ../Inc/stm32f10x_conf.h.
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
