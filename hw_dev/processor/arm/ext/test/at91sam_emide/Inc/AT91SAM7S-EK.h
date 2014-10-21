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
 *  File Name           : AT91SAM7S-EK.h
 *  Object              : AT91SAM7S-EK Evaluation Board Features Definition File
 *  Creation            : FDy   10-Nov-2006
 *-----------------------------------------------------------------------------
 */
#ifndef AT91SAM7S_EK_H
#define AT91SAM7S_EK_H


/*-----------------*/
/* LEDs Definition */
/*-----------------*/
#define AT91B_LED1          AT91C_PIO_PA0
#define AT91B_LED2          AT91C_PIO_PA1
#define AT91B_LED3          AT91C_PIO_PA2
#define AT91B_LED4          AT91C_PIO_PA3

#define AT91B_NB_LEB        4
#define AT91B_LED_MASK      AT91B_LED1 | AT91B_LED2 | AT91B_LED3 | AT91B_LED4
#define AT91D_BASE_PIO_LED  AT91C_BASE_PIOA
#define AT91D_ID_PIO_LED    AT91C_ID_PIOA

/*-------------------------------*/
/* BUTTONS Position Definition   */
/*-------------------------------*/
#define AT91B_BP1    AT91C_PIO_PA19
#define AT91B_BP2    AT91C_PIO_PA20
#define AT91B_BP3    AT91C_PIO_PA15
#define AT91B_BP4    AT91C_PIO_PA14

#define AT91B_BP_MASK       AT91B_BP1 | AT91B_BP2 | AT91B_BP3 | AT91B_BP4
#define AT91D_BASE_PIO_SW   AT91C_BASE_PIOA
#define AT91D_ID_PIO_SW     AT91C_ID_PIOA

#define AT91B_DBGU_BAUD_RATE	115200


/*---------------*/
/* Clocks       */
/*--------------*/
#define AT91B_MAIN_OSC        18432000               // Main Oscillator MAINCK
#define AT91B_MCK             ((18432000*73/14)/2)   // Output PLL Clock (48 MHz)
#define AT91B_MASTER_CLOCK     AT91B_MCK

#endif /* AT91SAM7S-EK_H */
