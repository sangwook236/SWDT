#include <AT91SAM7S-EK.h>
#include <AT91SAM7S256.h>
#include <cstdlib>


// Watchdog.
#define _WDT_BASE_ADDR  (0xFFFFFD40)
#define _WDT_MR         (*(volatile unsigned*) (_WDT_BASE_ADDR + 0x04))

#if defined(__cplusplus)
extern "C" {
#endif

// Sample implementation of __low_level_init().
// Disable watchdog on ATMEL AT91SAM7S.
// Gets called from startup code and has to return 1 to perform segment initialization.
int __low_level_init(void)
{
    _WDT_MR = (1 << 15);  // Initially disable watchdog.

    return 1;
}

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

#if defined(__cplusplus)
}
#endif

namespace my_at91sam7s {

// Send a string through the DBGU channel (very low level debugging).
void dbgu_print_ascii(const char *buffer);
// Print a 32-bit value in hexa.
void dbgu_print_hex8(unsigned long value);
// Delay.
void delay(unsigned long ms);

}  // namespace my_at91sam7s

namespace {
namespace local {

volatile unsigned long jiffies = 0;

// Data abort handler.
void dabt_handler()
{
	my_at91sam7s::dbgu_print_ascii("-F- Data abort at ");
	my_at91sam7s::dbgu_print_hex8(AT91C_BASE_MC->MC_AASR);
	my_at91sam7s::dbgu_print_ascii("\n\r");
}

// PITC interrupt handler.
void pitc_handler()
{
    unsigned long pivr = 0;
    unsigned long pisr = 0;

    // Read the PISR.
    pisr = AT91C_BASE_PITC->PITC_PISR & AT91C_PITC_PITS;
    if (0 != pisr)
    {
        // Read the PIVR. It acknowledges the IT.
        pivr = AT91C_BASE_PITC->PITC_PIVR;

        // Add to jiffies PICNT: the number of occurrences of periodic intervals since the last read of PIT_PIVR.
        jiffies += (pivr >> 20);
    }
}

// TC interrupt handler.
void timer_handler()
{
    // Clear status bit.
    volatile unsigned long dummy = AT91C_BASE_TC0->TC_SR;
/*
    // Toggle LED state.
    if (BASE_PIO_LED->PIO_ODSR & LED_B)
    {
        BASE_PIO_LED->PIO_CODR = LED_B;
    }
    else
    {
        BASE_PIO_LED->PIO_SODR = LED_B;
    }
*/
}

// Configure periodic interval timer.
void configure_pit()
{
    unsigned long pimr = 0;

    // Configure a resolution of 1 ms.
    AT91C_BASE_PITC->PITC_PIMR = AT91B_MASTER_CLOCK / (16 * 1000) - 1;

    // Enable interrupts.
    // Disable the interrupt on the interrupt controller.
    AT91C_BASE_AIC->AIC_IDCR = (1 << AT91C_ID_SYS);

    // Save the interrupt handler routine pointer and the interrupt priority.
    AT91C_BASE_AIC->AIC_SVR[AT91C_ID_SYS] = (unsigned long)pitc_handler;
    // Store the Source Mode Register.
    AT91C_BASE_AIC->AIC_SMR[AT91C_ID_SYS] = AT91C_AIC_SRCTYPE_INT_HIGH_LEVEL | AT91C_AIC_PRIOR_LOWEST;
    // Clear the interrupt on the interrupt controller.
    AT91C_BASE_AIC->AIC_ICCR = (1 << AT91C_ID_SYS);

    // Enable the interrupt on the interrupt controller.
    AT91C_BASE_AIC->AIC_IECR = (1 << AT91C_ID_SYS);

    // Enable the interrupt on the pit.
    pimr = AT91C_BASE_PITC->PITC_PIMR;
    AT91C_BASE_PITC->PITC_PIMR = pimr | AT91C_PITC_PITIEN;

    // Enable the pit.
    pimr = AT91C_BASE_PITC->PITC_PIMR;
    AT91C_BASE_PITC->PITC_PIMR = pimr | AT91C_PITC_PITEN;
}

// Configure timer/counter.
void configure_tc()
{
    volatile unsigned long dummy;

    // Enable periph clock for the PIO controller.
    AT91C_BASE_PMC->PMC_PCER = (1 << AT91C_ID_TC0);

    // Enable the periph.
    // Disable the clock and the interrupts.
    AT91C_BASE_TC0->TC_CCR = AT91C_TC_CLKDIS;
    AT91C_BASE_TC0->TC_IDR = 0xFFFFFFFF;

    // Clear status bit.
    dummy = AT91C_BASE_TC0->TC_SR;

    // Set the mode of the timer/counter.
    AT91C_BASE_TC0->TC_CMR = AT91C_TC_CLKS_TIMER_DIV5_CLOCK | AT91C_TC_CPCTRG;
    AT91C_BASE_TC0->TC_RC = AT91B_MASTER_CLOCK >> 12;  // MCKR divided by 1024 * 4.

    // Enable interrupts */
    // Disable the interrupt on the interrupt controller.
    AT91C_BASE_AIC->AIC_IDCR = (1 << AT91C_ID_TC0);
    // Save the interrupt handler routine pointer and the interrupt priority.
    AT91C_BASE_AIC->AIC_SVR[AT91C_ID_TC0] = (unsigned long)timer_handler;
    // Store the Source Mode Register.
    AT91C_BASE_AIC->AIC_SMR[AT91C_ID_TC0] = AT91C_AIC_SRCTYPE_INT_HIGH_LEVEL | AT91C_AIC_PRIOR_LOWEST;
    // Clear the interrupt on the interrupt controller.
    AT91C_BASE_AIC->AIC_ICCR = (1 << AT91C_ID_TC0);

    AT91C_BASE_TC0->TC_IER = AT91C_TC_CPCS;

    // Enable the interrupt on the interrupt controller.
    AT91C_BASE_AIC->AIC_IECR = (1 << AT91C_ID_TC0);

    // Enable the LED B timer.
/*
    if (led_b_active)
    {
      // Clock is started.
      AT91C_BASE_TC0->TC_CCR = AT91C_TC_CLKEN;
      // Counter is reset and the clock is started.
      AT91C_BASE_TC0->TC_CCR = AT91C_TC_SWTRG;
    }
*/
}

void configure_dbgu()
{
    // Reset and disable receiver.
    AT91C_BASE_DBGU->DBGU_CR = AT91C_US_RSTRX | AT91C_US_RSTTX;

    // Disable interrupts.
    AT91C_BASE_DBGU->DBGU_IDR = 0xFFFFFFFF;

    // Configure PIOs for DBGU.
    AT91C_BASE_PIOA->PIO_ASR = AT91C_PA9_DRXD | AT91C_PA10_DTXD;
    AT91C_BASE_PIOA->PIO_BSR = 0;
    AT91C_BASE_PIOA->PIO_PDR = AT91C_PA9_DRXD | AT91C_PA10_DTXD;

    // Configure serial link.
    // Define the baud rate divisor register [BRGR = MCK / (115200 * 16)].
    AT91C_BASE_DBGU->DBGU_BRGR = 26;
    // Define the USART mode.
    AT91C_BASE_DBGU->DBGU_MR = AT91C_US_PAR_NONE | AT91C_US_CHMODE_NORMAL;

    // Disable the RX and TX PDC transfer requests.
    AT91C_BASE_DBGU->DBGU_PTCR = AT91C_PDC_RXTDIS;
    AT91C_BASE_DBGU->DBGU_PTCR = AT91C_PDC_TXTDIS;

    // Enable transmitter.
    AT91C_BASE_DBGU->DBGU_CR = AT91C_US_TXEN;
}

}  // namespace local
}  // unnamed namespace

namespace my_at91sam7s {

// Send a string through the DBGU channel (very low level debugging).
void dbgu_print_ascii(const char *buffer)
{
    while ('\0' != *buffer)
    {
        while (!(AT91C_BASE_DBGU->DBGU_CSR & AT91C_US_TXRDY));
        AT91C_BASE_DBGU->DBGU_THR = (*buffer++ & 0x1FF);
    }
}

// Print a 32-bit value in hexa.
void dbgu_print_hex8(unsigned long value)
{
    char c = 0;
    char shift = sizeof(unsigned long) * 8;

    dbgu_print_ascii("0x");
    do
    {
        shift -= 4;
        while (!(AT91C_BASE_DBGU->DBGU_CSR & AT91C_US_TXRDY));
        c = ((value >> shift) & 0xF);
        if (c > 9)
            AT91C_BASE_DBGU->DBGU_THR = (('A' + (c - 10)) & 0x1FF);
        else
            AT91C_BASE_DBGU->DBGU_THR = (('0' + c) & 0x1FF);
    } while (shift != 0);
}

void delay(unsigned long ms)
{
    volatile unsigned long current_time = local::jiffies;
    unsigned long prev_jiffies;
    unsigned long target_time = current_time + ms;

    // Handle the counter overflow.
    if (target_time < current_time)
    {
        prev_jiffies = current_time;
        while (prev_jiffies <= local::jiffies)
            prev_jiffies = local::jiffies;
    }

    // Loop until the target time is reached.
    while (local::jiffies < target_time);
}

}  // namespace my_at91sam7s

int main(int argc, char *argv[])
{
    int gpio_main(int argc, char *argv[]);

 	int retval = EXIT_SUCCESS;
	{
        // FIXME [check] >> This project is not yet tested in H/W.

        // PITC configuration.
        local::configure_pit();
        // Timer/counter configuration.
        local::configure_tc();

        // DBGU configuration.
        local::configure_dbgu();

        retval = gpio_main(argc, argv);
	}

	return retval;
}
