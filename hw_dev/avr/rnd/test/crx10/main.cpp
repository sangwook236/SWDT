// [ref] ${PROCYON_AVRLIB_HOME}/encoder.h & encoder.c.
// [ref] ${PROCYON_AVRLIB_HOME}/conf/encoderconf.h.
#include "encoder.h"
#include <avr/io.h>
#include <avr/iom128.h>
#include <avr/interrupt.h>
#include <util/delay.h>
#include <stdlib.h>
#include <math.h>


#if 0

// interrupt service routine for external interrupt 0.
ISR(INT0_vect)
{
	PORTA = 0xE0;
	PORTC = PIND & 0xFE;
}

// interrupt service routine for external interrupt 1.
ISR(INT1_vect)
{
	PORTA = 0x07;
	PORTC = PIND & 0xFD;
}

// CRX10 encoder practice 1 (pp. 125~126).
int crx10_encoder_practice_main()
{
	DDRA = 0xFF;
	DDRC = 0xFF;
	DDRD = 0x00;
	
	EIMSK |= 0x03;
	EICRA |= 0x0F;
	EIFR |= 0x03;
	
	sei();
	
	while (true)
    {
        // TODO:: Please write your application code.
    }
	
	return 0;
}

#elif 0

// dot matrix practice (pp. 95~96).
int crx10_dot_matrix_practice_main()
{
	DDRA = 0xFF;
	DDRC = 0xFF;
	
	while (true)
	{
		PORTA = 0x02;
		PORTC = 0xFE;
		_delay_ms(1);
		PORTA = 0x04;
		PORTC = 0xFF;
		_delay_ms(1);
		PORTA = 0x08;
		PORTC = 0x1B;
		_delay_ms(1);
		PORTA = 0x10;
		PORTC = 0x1B;
		_delay_ms(1);
		PORTA = 0x20;
		PORTC = 0xFF;
		_delay_ms(1);
		PORTA = 0x40;
		PORTC = 0xFE;	
		_delay_ms(1);
	}
	
	return 0;
}

#elif 0

unsigned char u8_usart0_receive_data;

ISR(USART0_RX_vect)
{
	u8_usart0_receive_data = UDR0;
}

void usart0_init()
{
	// USART0, Baud rate: 115200 bps, Data bit: 8, Stop bit: 1, Parity: none.
	UBRR0H = 0;
	UBRR0L = 7;
	UCSR0A = (0 << RXC0) | (1 << UDRE0);
	UCSR0B = 0x98;
	UCSR0C = 0x06;
}

void usart0_transmit_data(unsigned char ch)
{
	while ((UCSR0A & 0x20) == 0x00);
	UDR0 = ch;
}

// USART practice (pp. 329~331).
int crx10_usart_practice_main()
{
	usart0_init();

	unsigned char usartdata = 'A';
	while (true)
	{
		usart0_transmit_data(usartdata);
		_delay_ms(100);
	}
	
	return 0;
}

#elif 0

// motor control practice (pp. 187~188).
int crx10_motor_control_practice_main()
{
	const unsigned char TOP = 200;  // TOP value.
	
	// initialize PWM.
	{
		DDRB = 0xF7;
		DDRE = 0xFB;
		DDRG = 0x00;
	
		// [ WGM13 WGM12 WGM11 WGM10 ] = 0b1110 (mode 14: Fast PWM).
		// [ COM1A1 COM1A0 ] = 0b10 (non-inverted PWM), [ COM1B1 COM1B0 ] = 0b10 (non-inverted PWM), [ COM1C1 COM1C0 ] = 0b00.
		// [ CS12 CS11 CS10 ] = 0b001 (Prescaler: F_CPU / 1).
		TCCR1A = 0xA2;
		TCCR1B = 0x19;
		// [ FOC1A FOC1B FOC1C ] = 0b000.
		TCCR1C = 0x00;
		// PWM's frequency = (F_CPU / Prescaler) / (1 + TOP).
		ICR1 = TOP;  // TOP value.
		OCR1A = 0;
		OCR1B = 0;
		OCR1C = 0;
/*
		// [ WGM33 WGM32 WGM31 WGM30 ] = 0b1110 (mode 14: Fast PWM).
		// [ COM3A1 COM3A0 ] = 0b11 (inverted PWM), [ COM3B1 COM3B0 ] = 0b00, [ COM3C1 COM3C0 ] = 0b00.
		// [ CS32 CS31 CS30 ] = 0b010 (Prescaler: F_CPU / 8).
		TCCR3A = 0xC2;
		TCCR3B = 0x1A;
		// [ FOC3A FOC3B FOC3C ] = 0b000.
		TCCR3C = 0x00;
		// PWM's frequency = (F_CPU / Prescaler) / (1 + TOP).
		ICR3 = 1000;  // TOP value.
		OCR3A = 500;
		OCR3B = 100;
		OCR3C = 0;
*/
	}

	// right wheel.
	PORTE &= 0x3F;  // PORTE7 = 0, PORTE6 = 0.
	PORTE |= 0x80;  // PORTE7 = 1 (forward).
	//PORTE |= 0x40;  // PORTE6 = 1 (backward).
	// left wheel.
	PORTG &= 0xE7;  // PORTG4 = 0, PORTG3 = 0 .
	PORTG |= 0x08;  // PORTG3 = 1 (forward).
	//PORTG |= 0x10;  // PORTG4 = 1 (backward).
	
	while (true)
	{
		// PWM's duty ratio = (1 + OCRnx) / (1 + TOP).
		OCR1A = round(TOP * 0.75);  // right wheel.
		OCR1B = round(TOP * 0.75);  // left wheel.
		
		_delay_ms(100);
	}
	
	return 0;
}

#elif 0

void usart0_init()
{
	// USART0, Baud rate: 115200 bps, Data bit: 8, Stop bit: 1, Parity: none.
	UBRR0H = 0;
	UBRR0L = 7;
	UCSR0A = (0 << RXC0) | (1 << UDRE0);
	UCSR0B = 0x98;
	UCSR0C = 0x06;
}

void usart0_transmit_data(unsigned char ch)
{
	while ((UCSR0A & 0x20) == 0x00);
	UDR0 = ch;
}

// CRX10 encoder interfacing.
int crx10_encoder_interfacing_main()
{
	// [ref] ${PROCYON_AVRLIB_HOME}/examples/encoder/encodertest.c.
		
	usart0_init();
	encoderInit();
	
	s32 div1 = 100000000, div2 = div1 / 10;
	while (true)
	{
		const s32 enc0 = encoderGetPosition(0);
		const s32 enc1 = encoderGetPosition(1);

		div1 = 100000000;
		div2 = div1 / 10;
		usart0_transmit_data(enc0 >= 0 ? '+' : '-');
		for (int i = 0; i < 8; ++i)
		{
			usart0_transmit_data((abs(enc0) % div1) / div2 + '0');
			div1 = div2;
			div2 /= 10;
		}
		usart0_transmit_data(',');
		usart0_transmit_data(' ');

		div1 = 100000000;
		div2 = div1 / 10;
		usart0_transmit_data(enc1 >= 0 ? '+' : '-');
		for (int i = 0; i < 8; ++i)
		{
			usart0_transmit_data((abs(enc1) % div1) / div2 + '0');
			div1 = div2;
			div2 /= 10;
		}
		usart0_transmit_data('\r');
		usart0_transmit_data('\n');
		
		_delay_ms(500);
	}
	
	return 0;
}

#elif 1

// CRX10 motor(wheel) control.
int crx10_motor_control_main()
{
	// gear ratio: 30 : 1.
	const int PULSES_PER_REV = 600;  // 60 [pulses/rev] (???).
	const unsigned char TOP = 200;  // TOP value.
	
	unsigned char ocr_value = TOP / 2;
	double duty = 0.0;  // duty ratio, [0, 1].

	const double ref_pos = M_PI_2;  // [rad].
	double curr_pos;  // [rad].
	double curr_err = 0.0, prev_err = 0.0, derr, sum_err = 0.0;  // [rad].

	const double Kp = 100.0, Kd = 25.0, Ki = 0.0;
	double pid_output = 0.0;

	// initialize PWM.
	{
		DDRB = 0xF7;
		DDRE = 0xFB;
		DDRG = 0x00;
	
		// [ WGM13 WGM12 WGM11 WGM10 ] = 0b1110 (mode 14: Fast PWM).
		// [ COM1A1 COM1A0 ] = 0b10 (non-inverted PWM), [ COM1B1 COM1B0 ] = 0b10 (non-inverted PWM), [ COM1C1 COM1C0 ] = 0b00.
		// [ CS12 CS11 CS10 ] = 0b001 (Prescaler: F_CPU / 1).
		TCCR1A = 0xA2;
		TCCR1B = 0x19;
		// [ FOC1A FOC1B FOC1C ] = 0b000.
		TCCR1C = 0x00;
		// PWM's frequency = (F_CPU / Prescaler) / (1 + TOP).
		ICR1 = TOP;  // TOP value.
		OCR1A = 0;
		OCR1B = 0;
		OCR1C = 0;
	}

	encoderInit();
	
	while (true)
	{
		// read encoder.
		const s32 enc0 = encoderGetPosition(0);
		//const s32 enc1 = encoderGetPosition(1);

		// pulses to radian.
		curr_pos = ((double)enc0 / (double)PULSES_PER_REV) * 2.0 * M_PI;  // [rad].

		curr_err = ref_pos - curr_pos;
		derr = curr_err - prev_err;
		sum_err = 0.0;  // no integral control.

		// PID control.
		pid_output = Kp * curr_err + Kd * derr + Ki * sum_err;
		
		// PID output to PWM's duty ratio.
		duty = fabs(pid_output) / 100.0;
		// saturation.
		if (duty < 0.0) duty = 0.0;
		else if (duty > 1.0) duty =1.0;

		// PWM's duty ratio = (1 + OCRnx) / (1 + TOP).
		//ocr_value = round(duty * (1 + TOP) - 1);
		ocr_value = round((duty + 1.0f) * TOP * 0.5f);  // 0 : duty : 1 = TOP/2 : OCRnx : TOP.
		
		// right wheel.
		PORTE &= 0x3F;  // PORTE7 = 0, PORTE6 = 0.
		if (curr_err >= 0.0)
			PORTE |= 0x80;  // PORTE7 = 1 (forward).
		else
			PORTE |= 0x40;  // PORTE6 = 1 (backward).
		
		OCR1A = ocr_value;  // right wheel.
		//OCR1B = ocr_value;  // left wheel.

		prev_err = curr_err;
		
		// control period : important !!!.
		_delay_ms(5);
		//_delay_ms(50);
	}
	
	return 0;
}

#endif

int main()
{
	//return crx10_encoder_practice_main();
	//return crx10_dot_matrix_practice_main();
	//return crx10_usart_practice_main();
	//return crx10_motor_control_practice_main();
	
	//return crx10_encoder_interfacing_main();
	return crx10_motor_control_main();
}
