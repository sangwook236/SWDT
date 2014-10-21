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
 *  File Name           : project.h
 *  Object              : project specific include file
 *  Creation            : FDy   10-Nov-2006
 *-----------------------------------------------------------------------------
 */
#ifndef _PROJECT_H
#define _PROJECT_H

/// Include your AT91 Library files and specific compiler definitions

#include "AT91SAM7S-EK.h"

#ifdef AT91SAM7S32
#include "AT91SAM7S32.h"
#endif
#ifdef AT91SAM7S321
#include "AT91SAM7S321.h"
#endif
#ifdef AT91SAM7S64
#include "AT91SAM7S64.h"
#endif
#ifdef AT91SAM7S128
#include "AT91SAM7S128.h"
#endif
#ifdef AT91SAM7S256
#include "AT91SAM7S256.h"
#endif

#ifndef __ASSEMBLY__
extern void dbgu_print_ascii(const char *buffer);
extern void dbgu_print_hex8(unsigned long);
#endif

#endif  // _PROJECT_H
