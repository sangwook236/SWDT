#include "usart.h"
#include <avr/interrupt.h>
#include <avr/sfr_defs.h>


#if defined(__cplusplus) && defined(__USE_CLASS_IN_USART)

class Usart0Buffer
{
public:
	Usart0Buffer()
	: rxBufLen_(64UL), txBufLen_(64UL),
	  rxBufStartIndex_(0), rxBufEndIndex_(0), txBufStartIndex_(0), txBufEndIndex_(0),
	  isTxBufFull_(0), isTransmitActive_(0)
	{
		for (int i = 0; i < rxBufLen_; ++i)
			rxBuf_[i] = 0;

		for (int i = 0; i < txBufLen_; ++i)
			txBuf_[i] = 0;
	}
	~Usart0Buffer()
	{}

private:
	Usart0Buffer(const Usart0Buffer &);  // Un-implemented
	Usart0Buffer & Usart0Buffer(const Usart0Buffer &);  // Un-implemented.

public:
	void reset()
	{
		// Empties reception buffers.
		rxBufStartIndex_ = rxBufEndIndex_ = 0;
		// Empties transmission buffers.
		txBufStartIndex_ = txBufEndIndex_ = 0;
		// Clears 'isTxBufFull_' & 'isTransmitActive_' flags.
		isTxBufFull_ = isTransmitActive_ = 0;
	}

	int8_t pushChar(const uint8_t ch)
	{
		if (!isTxBufFull_)  // Transmits only if buffer is not full.
		{
			if (!isTransmitActive_)  // If transmitter is not active.
			{
				// Transfers the first char direct to UDR0 to start transmission.
				isTransmitActive_ = 1;
				UDR0 = ch;
			}
			else
			{
				UCSR0B &= ~(_BV(TXCIE0));  // Disables USART0 TX complete interrupt after buffer is updated.
				txBuf_[txBufEndIndex_++ & (txBufLen_ - 1)] = ch;
				// Puts a char to transmission buffer.
				if (((txBufEndIndex_ ^ txBufStartIndex_) & (txBufLen_ - 1)) == 0)
				{
					isTxBufFull_ = 1;
				}
				UCSR0B |= _BV(TXCIE0);  // Enables USART0 TX complete interrupt after buffer is updated.
			}

			return 1;
		}
		else return 0;
	}

	void popChar()
	{
		if (rxBufEndIndex_ != rxBufStartIndex_)
		{
			UCSR0B &= ~(_BV(RXCIE0));  // Disables USART0 RX complete interrupt after buffer is updated.
			//ch = rxBuf_[++rxBufStartIndex_ & (rxBufLen_ - 1)];
			++rxBufStartIndex_;
			UCSR0B |= _BV(RXCIE0);  // Enables USART0 RX complete interrupt after buffer is updated.
		}
	}

	uint8_t topChar() const
	{
		uint8_t ch = 0xFF;

		if (rxBufEndIndex_ != rxBufStartIndex_)
		{
			UCSR0B &= ~(_BV(RXCIE0));  // Disables USART0 RX complete interrupt after buffer is updated.
			ch = rxBuf_[rxBufStartIndex_ & (rxBufLen_ - 1)];
			UCSR0B |= _BV(RXCIE0);  // Enables USART0 RX complete interrupt after buffer is updated.
		}

		return ch;
	}

	//
	int8_t isTxBufEmpty() const
	{  return txBufEndIndex_ == txBufStartIndex_;  }

	uint32_t getTxBufSize() const
	{
		return txBufEndIndex_ >= txBufStartIndex_ ? txBufEndIndex_ - txBufStartIndex_ : (uint32_t)((0x01 << sizeof(uint8_t)) - txBufStartIndex_ + txBufEndIndex_);
	}

	int8_t isRxBufEmpty() const
	{  return rxBufEndIndex_ == rxBufStartIndex_;  }

	uint32_t getRxBufSize() const
	{
		return rxBufEndIndex_ >= rxBufStartIndex_ ? rxBufEndIndex_ - rxBufStartIndex_ : (uint32_t)((0x01 << sizeof(uint8_t)) - rxBufStartIndex_ + rxBufEndIndex_);
	}

	void setTxBufFull(const int8_t flag)
	{  isTxBufFull_ = flag;  }
	const int8_t & isTxBufFull() const
	{  return isTxBufFull_;  }

	void setTransmitActive(const int8_t flag)
	{  isTransmitActive_ = flag;  }
	const int8_t & isTransmitActive() const
	{  return isTransmitActive_;  }

	//
	uint8_t getCharToTransmit()
	{
		return txBuf_[txBufStartIndex_++ & (txBufLen_ - 1)];
	}
	void setCharToBeReceived(const uint8_t ch)
	{
		rxBuf_[rxBufEndIndex_++ & (rxBufLen_ - 1)] = ch;
	}

public:
	const uint32_t rxBufLen_;
	const uint32_t txBufLen_;

private:
	// Start & end indexes of USARTn reception buffer.
	uint8_t rxBufStartIndex_, rxBufEndIndex_;
	// A storage for USARTn reception buffer.
	uint8_t rxBuf_[rxBufLen_];

	// Start & end indexes of USARTn transmission buffer.
	uint8_t txBufStartIndex_, txBufEndIndex_;
	// A storage for USARTn transmission buffer.
	uint8_t txBuf_[txBufLen_];

	// Marks a USARTn transmission buffer flag full.
	int8_t isTxBufFull_;
	// Marks a USARTn transmitter flag active.
	int8_t isTransmitActive_;
};

static volatile Usart0Buffer usart0Buf;

#else  // __cplusplus && __USE_CLASS_IN_USART

#define RxBufLen_Usart0 64  // A size of USART0 reception buffer.
static volatile uint8_t rxBufStartIndex_Usart0, rxBufEndIndex_Usart0;  // Start & end indexes of USART0 reception buffer.
static volatile uint8_t rxBuf_Usart0[RxBufLen_Usart0];  // A storage for USART0 reception buffer.

#define TxBufLen_Usart0 64  // A size of USART0 transmission buffer.
static volatile uint8_t txBufStartIndex_Usart0, txBufEndIndex_Usart0;  // Start & end indexes of USART0 transmission buffer.
static volatile uint8_t txBuf_Usart0[TxBufLen_Usart0];  // A storage for USART0 transmission buffer.

static volatile int8_t isTxBufFull_Usart0;  // Flag: marks a USART0 transmission buffer full.
static volatile int8_t isTransmitActive_Usart0;  // Flag: marks a USART0 transmitter active.

#endif  // __cplusplus && __USE_CLASS_IN_USART

ISR(USART0_TX_vect)
{
#if defined(__cplusplus) && defined(__USE_CLASS_IN_USART)

	// If chars are in a Tx buffer, transmits a char.
	if (!usart0Buf.isTxBufEmpty())
	{
		// TODO [check] >>
		UDR0 = getCharToTransmit();
		usart0Buf.setTxBufFull(0);
	}
	else usart0Buf.setTransmitActive(0);

#else  // __cplusplus && __USE_CLASS_IN_USART

	// If chars are in a Tx buffer, transmits a char.
	if (txBufStartIndex_Usart0 != txBufEndIndex_Usart0)
	{
		UDR0 = txBuf_Usart0[txBufStartIndex_Usart0++ & (TxBufLen_Usart0 - 1)];
		isTxBufFull_Usart0 = 0;  // clears 'isTxBufFull_Usart0' flag.
	}
	// If all chars are transmitted.
	else
	{
		isTransmitActive_Usart0 = 0;  // Clears 'isTransmitActive_Usart0' flag.
	}

#endif  // __cplusplus && __USE_CLASS_IN_USART
}

ISR(USART0_RX_vect)
{
	const uint8_t ch = UDR0;  // Reads a char.

#if defined(__cplusplus) && defined(__USE_CLASS_IN_USART)

	if (!usart0Buf.isRxBufFull())  // Rx buffer is not full.
	{
		// TODO [check] >>
		setCharToBeReceived(ch);
	}

#else  // __cplusplus && __USE_CLASS_IN_USART

	if (rxBufStartIndex_Usart0 + RxBufLen_Usart0 != rxBufEndIndex_Usart0)  // Rx buffer is not full.
	{
		rxBuf_Usart0[rxBufEndIndex_Usart0++ & (RxBufLen_Usart0 - 1)] = ch;  // Pushes a char into buffer.
	}

#endif  // __cplusplus && __USE_CLASS_IN_USART
}

ISR(USART0_UDRE_vect)
{
}

void usart0_init(const uint32_t baudrate)
{
	UDR0 = 0;

	UCSR0A = 0x00;
	//UCSR0A |= _BV(U2X0);  // Double the USART0 transmission speed enable.
	
	UCSR0B = 0x00;
	UCSR0B |= _BV(RXCIE0);  // USART0 RX complete interrupt enable.
	UCSR0B |= _BV(TXCIE0);  // USART0 TX complete interrupt enable.
	UCSR0B &= ~(_BV(UDRIE0));  // USART0 data register empty interrupt disable.
	UCSR0B |= _BV(RXEN0);  // USART0 receiver enable.
	UCSR0B |= _BV(TXEN0);  // USART0 transmitter enable.

	UCSR0C = 0x00;
	UCSR0C &= ~(_BV(UMSEL0));  // Asynchronous mode.
	UCSR0C &= ~(_BV(UPM01));  // No parity.
	UCSR0C &= ~(_BV(UPM00));  // No parity.
	UCSR0C &= ~(_BV(USBS0));  // 1 stop bit.

	// Data size: 8 bits.
	UCSR0B &= ~(_BV(UCSZ02));
	UCSR0C |= _BV(UCSZ01);
	UCSR0C |= _BV(UCSZ00);

	// Calculate Baud rate.
	//	1. Asynchronous normal mode (U2Xn = 0).
	//		baudrate = Fosc / (16 * (UBRRn + 1))
	//		UBRRn = Fosc / (16 * baudrate) - 1
	//	2. Asynchronous double speed mode (U2Xn = 1).
	//		baudrate = Fosc / (8 * (UBRRn + 1))
	//		UBRRn = Fosc / (8 * baudrate) - 1
	//	3. Synchronous master mode.
	//		baudrate = Fosc / (2 * (UBRRn + 1))
	//		UBRRn = Fosc / (2 * baudrate) - 1

	// Baud rate: 57600.
	//const uint32_t baudrate = 57600UL;
	const uint16_t ubrr0 = (uint16_t)(F_CPU / (16.0 * baudrate) - 1.0);
	UBRR0H = (uint8_t)((ubrr0 >> 8) & 0x00FF);
	UBRR0L = (uint8_t)(ubrr0 & 0x00FF);

#if defined(__cplusplus) && defined(__USE_CLASS_IN_USART)

	//usart0Buf.reset();

#else  // __cplusplus && __USE_CLASS_IN_USART

	usart0_init_buffer();

#endif  // __cplusplus && __USE_CLASS_IN_USART
}

int8_t usart0_push_char(const uint8_t ch)
{
#if defined(__cplusplus) && defined(__USE_CLASS_IN_USART)

	return usart0Buf.pushChar(ch);
	
#else  // __cplusplus && __USE_CLASS_IN_USART

	if (!isTxBufFull_Usart0)  // Transmits only if buffer is not full.
	{
		if (!isTransmitActive_Usart0)  // If transmitter is not active.
		{
			// Transfers the first char direct to UDR0 to start transmission.
			isTransmitActive_Usart0 = 1;
			UDR0 = ch;
		}
		else {
			UCSR0B &= ~(_BV(TXCIE0));  // Disables USART0 TX complete interrupt after buffer is updated.
			txBuf_Usart0[txBufEndIndex_Usart0++ & (TxBufLen_Usart0 - 1)] = ch;
			// Puts a char to transmission buffer.
			if (((txBufEndIndex_Usart0 ^ txBufStartIndex_Usart0) & (TxBufLen_Usart0 - 1)) == 0)
			{
				isTxBufFull_Usart0 = 1;
			}
			UCSR0B |= _BV(TXCIE0);  // Enables USART0 TX complete interrupt after buffer is updated.
		}

		return 1;
	}
	else return 0;

#endif  // __cplusplus && __USE_CLASS_IN_USART
}

void usart0_pop_char()
{
#if defined(__cplusplus) && defined(__USE_CLASS_IN_USART)

	usart0Buf.popChar();
	
#else  // __cplusplus && __USE_CLASS_IN_USART

	if (rxBufEndIndex_Usart0 != rxBufStartIndex_Usart0)
	{
		UCSR0B &= ~(_BV(RXCIE0));  // Disables USART0 RX complete interrupt after buffer is updated.
		//ch = rxBuf_Usart0[++rxBufStartIndex_Usart0 & (RxBufLen_Usart0 - 1)];
		++rxBufStartIndex_Usart0;
		UCSR0B |= _BV(RXCIE0);  // Enables USART0 RX complete interrupt after buffer is updated.
	}
	
#endif  // __cplusplus && __USE_CLASS_IN_USART
}

uint8_t usart0_top_char()
{
#if defined(__cplusplus) && defined(__USE_CLASS_IN_USART)

	return usart0Buf.topChar();

#else  // __cplusplus && __USE_CLASS_IN_USART

	uint8_t ch = 0xFF;

	if (rxBufEndIndex_Usart0 != rxBufStartIndex_Usart0)
	{
		UCSR0B &= ~(_BV(RXCIE0));  // Disables USART0 RX complete interrupt after buffer is updated.
		ch = rxBuf_Usart0[rxBufStartIndex_Usart0 & (RxBufLen_Usart0 - 1)];
		UCSR0B |= _BV(RXCIE0);  // Enables USART0 RX complete interrupt after buffer is updated.
	}

	return ch;

#endif  // __cplusplus && __USE_CLASS_IN_USART
}

int8_t usart0_is_empty()
{
#if defined(__cplusplus) && defined(__USE_CLASS_IN_USART)

	return usart0Buf.isRxBufEmpty();

#else  // __cplusplus && __USE_CLASS_IN_USART

	return rxBufEndIndex_Usart0 == rxBufStartIndex_Usart0;

#endif  // __cplusplus && __USE_CLASS_IN_USART
}

uint32_t usart0_get_size()
{
#if defined(__cplusplus) && defined(__USE_CLASS_IN_USART)

	return usart0Buf.getRxBufSize();

#else  // __cplusplus && __USE_CLASS_IN_USART

	return rxBufEndIndex_Usart0 >= rxBufStartIndex_Usart0 ? rxBufEndIndex_Usart0 - rxBufStartIndex_Usart0 : (uint32_t)((0x01 << sizeof(uint8_t)) - rxBufStartIndex_Usart0 + rxBufEndIndex_Usart0);

#endif  // __cplusplus && __USE_CLASS_IN_USART
}

void usart0_init_buffer()
{
#if defined(__cplusplus) && defined(__USE_CLASS_IN_USART)

	usart0Buf.reset();

#else  // __cplusplus && __USE_CLASS_IN_USART

	// Empties reception buffers.
	rxBufStartIndex_Usart0 = rxBufEndIndex_Usart0 = 0;
	// Empties transmission buffers.
	txBufStartIndex_Usart0 = txBufEndIndex_Usart0 = 0;
	// Clears 'isTxBufFull_Usart0' & 'isTransmitActive_Usart0' flags.
	isTxBufFull_Usart0 = isTransmitActive_Usart0 = 0;

	for (int i = 0; i < RxBufLen_Usart0; ++i)
		rxBuf_Usart0[i] = 0;

	for (int i = 0; i < RxBufLen_Usart0; ++i)
		txBuf_Usart0[i] = 0;

#endif  // __cplusplus && __USE_CLASS_IN_USART
}
