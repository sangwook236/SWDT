/* ----------------------------------------------------------------------------
 *         ATMEL Microcontroller Software Support  -  ROUSSET  -
 * ----------------------------------------------------------------------------
 * Copyright (c) 2006, Atmel Corporation
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the disclaiimer below.
 *
 * - Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the disclaimer below in the documentation and/or
 * other materials provided with the distribution.
 *
 * Atmel's name may not be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * DISCLAIMER: THIS SOFTWARE IS PROVIDED BY ATMEL "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT ARE
 * DISCLAIMED. IN NO EVENT SHALL ATMEL BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * ----------------------------------------------------------------------------
 */
/*-----------------------------------------------------------------------------
 *  File Name           : main.c
 *  Object              : main file
 *  Creation            : FDy   10-Nov-2006
 *-----------------------------------------------------------------------------
 */

/* Include Standard files */
#include "project.h"

/* Global variables */
#define LED_A          AT91B_LED1
#define LED_B          AT91B_LED2
#define BASE_PIO_LED   AT91D_BASE_PIO_LED
#define ID_PIO_LED     AT91D_ID_PIO_LED
#define BUTTON_A       AT91B_BP1
#define BUTTON_B       AT91B_BP2
#define BASE_PIO_BP    AT91D_BASE_PIO_SW
#define ID_PIO_BP      AT91D_ID_PIO_SW

#define DEBOUNCE_TIME  500

/* The following global variables, control the LED blinking */
unsigned long led_a_active = 1;
unsigned long led_b_active = 1;
volatile unsigned long jiffies = 0;

static void configure_pit(void);
static void configure_buttons(void);
static void configure_leds(void);
static void configure_tc(void);
static void configure_dbgu(void);


/*-----------------------------------------------------------------------------
 * Function Name       : pabt_handler
 * Object              : Prefetch Abort Handler
 *-----------------------------------------------------------------------------*/
void pabt_handler(void)
{
	dbgu_print_ascii("-F- Prefetch abort at ");
	dbgu_print_hex8(AT91C_BASE_MC->MC_AASR);
	dbgu_print_ascii("\n\r");
}

/*-----------------------------------------------------------------------------
 * Function Name       : dabt_handler
 * Object              : Data Abort Handler
 *-----------------------------------------------------------------------------*/
void dabt_handler(void)
{
	dbgu_print_ascii("-F- Data abort at ");
	dbgu_print_hex8(AT91C_BASE_MC->MC_AASR);
	dbgu_print_ascii("\n\r");
}

/*-----------------------------------------------------------------------------
 * Function Name       : pitc_handler
 * Object              : Handler for PITC interrupt
 *-----------------------------------------------------------------------------*/
void pitc_handler(void)
{
  unsigned long pivr = 0;
  unsigned long pisr = 0;

  /* Read the PISR */
  pisr = AT91C_BASE_PITC->PITC_PISR & AT91C_PITC_PITS;
  if (pisr != 0) {
    /* Read the PIVR. It acknowledges the IT */
    pivr = AT91C_BASE_PITC->PITC_PIVR;

    /* Add to jiffies PICNT: the number of occurrences of periodic intervals  */
    /* since the last read of PIT_PIVR */
    jiffies += (pivr >> 20);
  }
}


/*-----------------------------------------------------------------------------
 * Function Name       : configure_pit
 * Object              : Configure priodic interval timer
 *-----------------------------------------------------------------------------*/
static void configure_pit(void)
{
  unsigned long pimr = 0;

  /* Configure a resolution of 1 ms */
  AT91C_BASE_PITC->PITC_PIMR = AT91B_MASTER_CLOCK / (16 * 1000) - 1;

  /* Enable interrupts */
  /* Disable the interrupt on the interrupt controller */
  AT91C_BASE_AIC->AIC_IDCR = (1 << AT91C_ID_SYS);

  /* Save the interrupt handler routine pointer and the interrupt priority */
  AT91C_BASE_AIC->AIC_SVR[AT91C_ID_SYS] = (unsigned long) pitc_handler;
  /* Store the Source Mode Register */
  AT91C_BASE_AIC->AIC_SMR[AT91C_ID_SYS] = AT91C_AIC_SRCTYPE_INT_HIGH_LEVEL | AT91C_AIC_PRIOR_LOWEST;
  /* Clear the interrupt on the interrupt controller */
  AT91C_BASE_AIC->AIC_ICCR = (1 << AT91C_ID_SYS);

  /* Enable the interrupt on the interrupt controller */
  AT91C_BASE_AIC->AIC_IECR = (1 << AT91C_ID_SYS);

  /* Enable the interrupt on the pit */
  pimr = AT91C_BASE_PITC->PITC_PIMR;
  AT91C_BASE_PITC->PITC_PIMR = pimr | AT91C_PITC_PITIEN;

  /* Enable the pit */
  pimr = AT91C_BASE_PITC->PITC_PIMR;
  AT91C_BASE_PITC->PITC_PIMR = pimr | AT91C_PITC_PITEN;
}


/*-----------------------------------------------------------------------------
 * Function Name       : button_handler
 * Object              : Button Interrupt Service Routine (PIO A)
 *-----------------------------------------------------------------------------*/
void button_handler(void)
{
  /* Read the Interrupt Status register (It acknowledge the IT)  */
  unsigned long pio_isr = BASE_PIO_BP->PIO_ISR;
  unsigned long pio_pdsr = BASE_PIO_BP->PIO_PDSR;
  static unsigned long button_a_jiffies = 0;
  static unsigned long button_b_jiffies = 0;


  /* If the BUTTON_A has been pushed or released */
  if (pio_isr & BUTTON_A) {
    /* Enable/disable the LED A blinking when BUTTON_A is released */
    if (pio_pdsr & BUTTON_A) {
      if (jiffies - button_a_jiffies > DEBOUNCE_TIME) {
	button_a_jiffies = jiffies;
	led_a_active = !led_a_active;
      }
    }
  }
  /* If the BUTTON_B has been pushed or released */
  else if (pio_isr & BUTTON_B) {
    /* Enable/disable the LED B blinking when BUTTON_B is pushed */
    if (!(pio_pdsr & BUTTON_B)) {
      if (jiffies - button_b_jiffies > DEBOUNCE_TIME) {
	button_b_jiffies = jiffies;
	if (led_b_active) {
	  /* Disable the clock */
	  AT91C_BASE_TC0->TC_CCR = AT91C_TC_CLKDIS;
	} else {
	  /* Clock is started */
	  AT91C_BASE_TC0->TC_CCR = AT91C_TC_CLKEN;
	  /* Counter is reset and the clock is started */
	  AT91C_BASE_TC0->TC_CCR = AT91C_TC_SWTRG;
	}
	led_b_active = !led_b_active;
      }
    }
  }
  /* The else should not be executed since no other PIO are enabled  */
  else { }
}


/*-----------------------------------------------------------------------------
 * Function Name       : configure_buttons
 * Object              : Configure pios to capture buttons state
 *-----------------------------------------------------------------------------*/
static void configure_buttons(void)
{
  /* Enable the periph clock for the PIO controller */
  /* This is mandatory when PIO are configured as input */
  AT91C_BASE_PMC->PMC_PCER = (1 << ID_PIO_BP);

  /* Set the PIO line in input */
  BASE_PIO_BP->PIO_ODR = (BUTTON_A | BUTTON_B);

  /* Set the PIO controller in PIO mode instead of peripheral mode */
  BASE_PIO_BP->PIO_PER = (BUTTON_A | BUTTON_B);

  /* Disable the interrupt on the interrupt controller */
  AT91C_BASE_AIC->AIC_IDCR = (1 << ID_PIO_BP);

  /* Save the interrupt handler routine pointer and the interrupt priority */
  AT91C_BASE_AIC->AIC_SVR[ID_PIO_BP] = (unsigned long) button_handler;

  /* Store the Source Mode Register */
  AT91C_BASE_AIC->AIC_SMR[ID_PIO_BP] = AT91C_AIC_SRCTYPE_INT_HIGH_LEVEL | AT91C_AIC_PRIOR_LOWEST;

  /* Clear the interrupt on the interrupt controller */
  AT91C_BASE_AIC->AIC_ICCR = (1 << ID_PIO_BP);

  /* Enable button interrupts generation through the PIO controller */
  BASE_PIO_BP->PIO_IER = (BUTTON_A | BUTTON_B);

  /* Enable PIO interrupt in the interrupt controller */
  AT91C_BASE_AIC->AIC_IECR = (1 << ID_PIO_BP);
}


/*-----------------------------------------------------------------------------
 * Function Name       : configure_leds
 * Object              : Configure pios to control led states
 *-----------------------------------------------------------------------------*/
static void configure_leds(void)
{
  /* Configure the pin in output */
  BASE_PIO_LED->PIO_OER = (LED_A | LED_B);
  /* Set the PIO controller in PIO mode instead of peripheral mode */
  BASE_PIO_LED->PIO_PER = (LED_A | LED_B);
  /* Disable pull-up */
  BASE_PIO_LED->PIO_PPUDR = (LED_A | LED_B);

  /* Set the default state for the led */
  if (led_a_active)
    BASE_PIO_LED->PIO_SODR = LED_A;
  else
    BASE_PIO_LED->PIO_CODR = LED_A;

  if (led_b_active)
    BASE_PIO_LED->PIO_SODR = LED_B;
  else
    BASE_PIO_LED->PIO_CODR = LED_B;
}


/*-----------------------------------------------------------------------------
 * Function Name       : timer_handler
 * Object              : Handler for TC interrupt
 *-----------------------------------------------------------------------------*/
void timer_handler(void)
{
  volatile unsigned long dummy;
  /* Clear status bit */
  dummy = AT91C_BASE_TC0->TC_SR;

  /* Toggle LED state */
  if (BASE_PIO_LED->PIO_ODSR & LED_B) {
    BASE_PIO_LED->PIO_CODR = LED_B;
  } else {
    BASE_PIO_LED->PIO_SODR = LED_B;
  }
}


/*-----------------------------------------------------------------------------
 * Function Name       : configure_tc
 * Object              : Configure TC
 *-----------------------------------------------------------------------------*/
static void configure_tc(void)
{
    volatile unsigned long dummy;

    /* Enable periph clock for the PIO controller */
    AT91C_BASE_PMC->PMC_PCER = (1 << AT91C_ID_TC0);

    /* Enable the periph */
    /* Disable the clock and the interrupts */
    AT91C_BASE_TC0->TC_CCR = AT91C_TC_CLKDIS;
    AT91C_BASE_TC0->TC_IDR = 0xFFFFFFFF;

    /* Clear status bit */
    dummy = AT91C_BASE_TC0->TC_SR;

    /* Set the Mode of the Timer Counter */
    AT91C_BASE_TC0->TC_CMR = AT91C_TC_CLKS_TIMER_DIV5_CLOCK | AT91C_TC_CPCTRG;
    AT91C_BASE_TC0->TC_RC = AT91B_MASTER_CLOCK >> 12;  /* MCKR divided by 1024 * 4 */

    /* Enable interrupts */
    /* Disable the interrupt on the interrupt controller */
    AT91C_BASE_AIC->AIC_IDCR = (1 << AT91C_ID_TC0);
    /* Save the interrupt handler routine pointer and the interrupt priority */
    AT91C_BASE_AIC->AIC_SVR[AT91C_ID_TC0] = (unsigned long) timer_handler;
    /* Store the Source Mode Register */
    AT91C_BASE_AIC->AIC_SMR[AT91C_ID_TC0] = AT91C_AIC_SRCTYPE_INT_HIGH_LEVEL | AT91C_AIC_PRIOR_LOWEST;
    /* Clear the interrupt on the interrupt controller */
    AT91C_BASE_AIC->AIC_ICCR = (1 << AT91C_ID_TC0);

    AT91C_BASE_TC0->TC_IER = AT91C_TC_CPCS;

    /* Enable the interrupt on the interrupt controller */
    AT91C_BASE_AIC->AIC_IECR = (1 << AT91C_ID_TC0);

    /* Enable the LED B timer */
    if (led_b_active) {
      /* Clock is started */
      AT91C_BASE_TC0->TC_CCR = AT91C_TC_CLKEN;
      /* Counter is reset and the clock is started */
      AT91C_BASE_TC0->TC_CCR = AT91C_TC_SWTRG;
    }
}


/*----------------------------------------------------------------------------
 * Function Name       : dbgu_print_ascii
 * Object              : This function is used to send a string through the
 *                       DBGU channel (Very low level debugging)
 *----------------------------------------------------------------------------*/
void dbgu_print_ascii(const char *buffer)
{
    while (*buffer != '\0') {
        while (!(AT91C_BASE_DBGU->DBGU_CSR & AT91C_US_TXRDY));
        AT91C_BASE_DBGU->DBGU_THR = (*buffer++ & 0x1FF);
    }
}


/*----------------------------------------------------------------------------
 * Function Name       : dbgu_print_hex8
 * Object              : This function is used to print a 32-bit value in hexa
 *----------------------------------------------------------------------------*/
void dbgu_print_hex8(unsigned long value)
{
    char c = 0;
    char shift = sizeof(unsigned long) * 8;

    dbgu_print_ascii("0x");
    do {
        shift -= 4;
        while (!(AT91C_BASE_DBGU->DBGU_CSR & AT91C_US_TXRDY));
        c = ((value >> shift) & 0xF);
        if (c > 9)
	  AT91C_BASE_DBGU->DBGU_THR = (('A' + (c - 10)) & 0x1FF);
        else
	  AT91C_BASE_DBGU->DBGU_THR = (('0' + c) & 0x1FF);
    } while (shift != 0);
}


/*-----------------------------------------------------------------------------
 * Function Name       : configure_dbgu
 * Object              : Configure DBGU
 *-----------------------------------------------------------------------------*/
static void configure_dbgu (void)
{
    /* Reset and disable receiver */
    AT91C_BASE_DBGU->DBGU_CR = AT91C_US_RSTRX | AT91C_US_RSTTX;

    /* Disable interrupts */
    AT91C_BASE_DBGU->DBGU_IDR = 0xFFFFFFFF;

    /* Configure PIOs for DBGU */
    AT91C_BASE_PIOA->PIO_ASR = AT91C_PA9_DRXD | AT91C_PA10_DTXD;
    AT91C_BASE_PIOA->PIO_BSR = 0;
    AT91C_BASE_PIOA->PIO_PDR = AT91C_PA9_DRXD | AT91C_PA10_DTXD;

    /* === Configure serial link === */
    /* Define the baud rate divisor register [BRGR = MCK / (115200 * 16)] */
    AT91C_BASE_DBGU->DBGU_BRGR = 26;
    /* Define the USART mode */
    AT91C_BASE_DBGU->DBGU_MR = AT91C_US_PAR_NONE | AT91C_US_CHMODE_NORMAL;

    /* Disable the RX and TX PDC transfer requests */
    AT91C_BASE_DBGU->DBGU_PTCR = AT91C_PDC_RXTDIS;
    AT91C_BASE_DBGU->DBGU_PTCR = AT91C_PDC_TXTDIS;

    /* Enable transmitter */
    AT91C_BASE_DBGU->DBGU_CR = AT91C_US_TXEN;
}


/*-----------------------------------------------------------------------------
 * Function Name       : wait
 * Object              : Tempo using jiffies (updated by PITC)
 *-----------------------------------------------------------------------------*/
void wait (unsigned long ms)
{
    volatile unsigned long current_time = jiffies;
    unsigned long prev_jiffies;
    unsigned long target_time = current_time + ms;

    /* Handle the counter overflow */
    if (target_time < current_time) {
        prev_jiffies = current_time;
        while (prev_jiffies <= jiffies)
            prev_jiffies = jiffies;
    }
    /* Loop until the target time is reached */
    while (jiffies < target_time);
}


/*-----------------------------------------------------------------------------
 * Function Name       : Main
 * Object              : Software entry point
 *-----------------------------------------------------------------------------*/
int main(void)
{
    /* ==== PITC configuration ==== */
    configure_pit();

    /* ==== Timer Counter configuration ==== */
    configure_tc();

    /* === BUTTON configuration (PIO in INPUT) === */
    configure_buttons();

    /* === LED configuration (PIO in OUTPUT) === */
    configure_leds();

    /* ==== DBGU configuration ==== */
    configure_dbgu();

    dbgu_print_ascii("AT91SAM7S Getting Started program launched ...\n\r");
    dbgu_print_ascii("AT91SAM7S chip ID : ");
    dbgu_print_hex8(AT91C_BASE_DBGU->DBGU_CIDR);
    dbgu_print_ascii("\n\r");

    while (1) {
        if (led_a_active) {
            /* Switch on the led */
            wait(500);
            BASE_PIO_LED->PIO_CODR = LED_A;

            /* Switch off the led */
            wait(500);
            BASE_PIO_LED->PIO_SODR = LED_A;
        }
    }

    return 0;
}
