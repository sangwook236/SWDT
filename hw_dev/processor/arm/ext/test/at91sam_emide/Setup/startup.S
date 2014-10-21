/************************************************************************************
 * File:    startup.S
 * Purpose: Startup file for ARM7/ARM7TDMI/ARM7TDMI-S devices. 
 * Date:    05 July 2013
 * PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 *   INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 *   IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 *   DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 *   TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
 *   OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 ************************************************************************************/
 
/************************************************************************************
 *
 *  Defines, used for the processor status register
 ************************************************************************************/
        ARM_MODE_USER  = 0x10        /* Normal User Mode                             */
        ARM_MODE_FIQ   = 0x11        /* FIQ Fast Interrupts Mode                     */
        ARM_MODE_IRQ   = 0x12        /* IRQ Standard Interrupts Mode                 */
        ARM_MODE_SVC   = 0x13        /* Supervisor Interrupts Mode                   */
        ARM_MODE_ABORT = 0x17        /* Abort Processing memory Faults Mode          */
        ARM_MODE_UNDEF = 0x1B        /* Undefined Instructions Mode                  */
        ARM_MODE_SYS   = 0x1F        /* System Running in Priviledged Operating Mode */
        ARM_MODE_MASK  = 0x1F

        I_BIT          = 0x80        /* Disables IRQ when I bit is set               */
        F_BIT          = 0x40        /* Disables FIQ when F bit is set               */

/************************************************************************************
 *
 *  The minimal vector table for ARM7.
 *  Device specific external interrupts can be added below.
 ************************************************************************************/
        .text
        .global  __vector
        .extern  Reset_Handler
         
        .arm
        .section .vectors, "ax" 
                        
__vector:
        ldr     pc,Reset_Addr   /* RESET                 vector */
        ldr     pc,Undef_Addr   /* Undefined instruction vector */
        ldr     pc,SWI_Addr     /* Software interrupt    vector */
        ldr     pc,PAbt_Addr    /* Prefetch abort        vector */
        ldr     pc,DAbt_Addr    /* Data abort            vector */
        nop                     /* Reserved              vector */
        ldr     pc,IRQ_Addr     /* Interrupt             vector */
        ldr     pc,FIQ_Addr     /* Fast interrupt        vector */

Reset_Addr:     .word   Reset_Handler
Undef_Addr:     .word   Undef_Handler
SWI_Addr:       .word   SWI_Handler
PAbt_Addr:      .word   PAbt_Handler
DAbt_Addr:      .word   DAbt_Handler
IRQ_Addr:       .word   IRQ_Handler
FIQ_Addr:       .word   FIQ_Handler

Undef_Handler:  b       Undef_Handler
SWI_Handler:    b       SWI_Handler
PAbt_Handler:   b       PAbt_Handler
DAbt_Handler:   b       DAbt_Handler
IRQ_Handler:    b       IRQ_Handler 
FIQ_Handler:    b       FIQ_Handler 
__vector_end:

/************************************************************************************
 *
 *  Standard C (crt0) initialization function
 *  Calls Low_level_init, copies data from ROM to RAM, clears BSS,
 *  finally calls main().
 ************************************************************************************/
        .extern  __low_level_init
        .extern  main

crt0:
        /*
         * Call __low_level_init to initiliaze hardware
         * before calling c-standard startup
         */
        ldr     r0,=__low_level_init
        mov     lr, pc
        bx      r0
        /*
         * Relocate .data section
         * (Copy from ROM to RAM)
         */
        ldr   r1, =__etext
        ldr   r2, =__data_start__
        ldr   r3, =__data_end__
LoopRel:
        cmp   r2, r3
        ldrlo r0, [r1], #4
        strlo r0, [r2], #4
        blo   LoopRel

        /*
         * Clear .bss section
         */
        ldr   r1, =__bss_start__
        ldr   r2, =__bss_end__
        ldr   r3, =0
bss_clear_loop:
        cmp   r1, r2
        strne r3, [r1], #+4
        bne   bss_clear_loop
        /*
         *  Prepare and call main()
         */
        mrs   r0, cpsr
        bic   r0, r0, #(I_BIT | F_BIT)     /* Enable FIQ and IRQ interrupt */
        msr   cpsr, r0
        mov   r0, #0                         /* No arguments are passed to main */
        mov   r1, #0
        ldr   r2, =main
        mov   lr, pc
        bx    r2
_exit:  b     _exit                          /* We should never come to here, just for sureness. */

/*********************************************************************
*
*       __low_level_init
*
**********************************************************************
*/
__low_level_init:
        bx lr
        .weak __low_level_init
/************************************************************************************
 *
 *  Reset_Handler()
 *  This function gets called at start of execution after a reset event.
 *  After a reset, the mode is ARM, Supervisor, interrupts disabled.
 ************************************************************************************/
        .global  Reset_Handler
        .global  end
        .arm
        .section .text, "ax"

Reset_Handler:
        /*
         * Setup a stack for each mode
         */
        msr   CPSR_c, #ARM_MODE_UNDEF | I_BIT | F_BIT   /* Undefined Instruction Mode */
        ldr   sp, =__stack_und_end__

        msr   CPSR_c, #ARM_MODE_ABORT | I_BIT | F_BIT   /* Abort Mode */
        ldr   sp, =__stack_abt_end__

        msr   CPSR_c, #ARM_MODE_FIQ   | I_BIT | F_BIT   /* FIQ Mode */
        ldr   sp, =__stack_fiq_end__

        msr   CPSR_c, #ARM_MODE_IRQ   | I_BIT | F_BIT   /* IRQ Mode */
        ldr   sp, =__stack_irq_end__

        msr   CPSR_c, #ARM_MODE_SVC   | I_BIT | F_BIT   /* Supervisor Mode */
        ldr   sp, =__stack_svc_end__

        /*
         * Now enter crt0 function,
         * which does low-level and segment initialization.
         * and then calls main().
         */
        ldr   r0, =crt0
        mov   lr, pc
        bx    r0
end:    b     end
        .end
