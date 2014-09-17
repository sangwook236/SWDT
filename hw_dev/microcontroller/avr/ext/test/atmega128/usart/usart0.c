#include "usart.h"
#include <avr/interrupt.h>
#include <avr/sfr_defs.h>


#if defined(__cplusplus)
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
	Usart0Buffer(const Usart0Buffer &);  // un-implemented
	Usart0Buffer & Usart0Buffer(const Usart0Buffer &);  // un-implemented

public:
	void reset()
	{
		// empties reception buffers
		rxBufStartIndex_ = rxBufEndIndex_ = 0;
		// empties transmission buffers
		txBufStartIndex_ = txBufEndIndex_ = 0;
		// clears 'isTxBufFull_' & 'isTransmitActive_' flags
		isTxBufFull_ = isTransmitActive_ = 0;
	}

	int8_t pushChar(const uint8_t ch)
	{
		if (!isTxBufFull_)  // transmits only if buffer is not full
		{
			if (!isTransmitActive_)  // if transmitter is not active
			{
				// transfers the first char direct to UDR0 to start transmission
				isTransmitActive_ = 1;
				UDR0 = ch;
			}
			else
			{
				UCSR0B &= ~(_BV(TXCIE0));  // disables USART0 TX complete interrupt after buffer is updated
				txBuf_[txBufEndIndex_++ & (txBufLen_ - 1)] = ch;
				// puts a char to transmission buffer
				if (((txBufEndIndex_ ^ txBufStartIndex_) & (txBufLen_ - 1)) == 0)
				{
					isTxBufFull_ = 1;
				}
				UCSR0B |= _BV(TXCIE0);  // enables USART0 TX complete interrupt after buffer is updated
			}

			return 1;
		}
		else return 0;
	}

	void popChar()
	{
		if (rxBufEndIndex_ != rxBufStartIndex_)
		{
			UCSR0B &= ~(_BV(RXCIE0));  // disables USART0 RX complete interrupt after buffer is updated
			//ch = rxBuf_[++rxBufStartIndex_ & (rxBufLen_ - 1)];
			++rxBufStartIndex_;
			UCSR0B |= _BV(RXCIE0);  // enables USART0 RX complete interrupt after buffer is updated
		}
	}

	uint8_t topChar() const
	{
		uint8_t ch = 0xFF;

		if (rxBufEndIndex_ != rxBufStartIndex_)
		{
			UCSR0B &= ~(_BV(RXCIE0));  // disables USART0 RX complete interrupt after buffer is updated
			ch = rxBuf_[rxBufStartIndex_ & (rxBufLen_ - 1)];
			UCSR0B |= _BV(RXCIE0);  // enables USART0 RX complete interrupt after buffer is updated
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
	// start & end indexes of USARTn reception buffer
	uint8_t rxBufStartIndex_, rxBufEndIndex_;
	// a storage for USARTn reception buffer
	uint8_t rxBuf_[rxBufLen_];

	// start & end indexes of USARTn transmission buffer
	uint8_t txBufStartIndex_, txBufEndIndex_;
	// a storage for USARTn transmission buffer
	uint8_t txBuf_[txBufLen_];

	// marks a USARTn transmission buffer flag full
	int8_t isTxBufFull_;
	// marks a USARTn transmitter flag active
	int8_t isTransmitActive_;
};

static volatile Usart0Buffer usart0Buf;
#else  // __cplusplus
#define RxBufLen_Usart0 64  // a size of USART0 reception buffer
static volatile uint8_t rxBufStartIndex_Usart0, rxBufEndIndex_Usart0;  // start & end indexes of USART0 reception buffer
static volatile uint8_t rxBuf_Usart0[RxBufLen_Usart0];  // a storage for USART0 reception buffer

#define TxBufLen_Usart0 64  // a size of USART0 transmission buffer
static volatile uint8_t txBufStartIndex_Usart0, txBufEndIndex_Usart0;  // start & end indexes of USART0 transmission buffer
static volatile uint8_t txBuf_Usart0[TxBufLen_Usart0];  // a storage for USART0 transmission buffer

static volatile int8_t isTxBufFull_Usart0;  // flag: marks a USART0 transmission buffer full
static volatile int8_t isTransmitActive_Usart0;  // flag: marks a USART0 transmitter active
#endif  // __cplusplus

ISR(USART0_TX_vect)
{
#if defined(__cplusplus)
	// if chars are in a Tx buffer, transmits a char
	if (!usart0Buf.isTxBufEmpty())
	{
		// TODO [check] >>
		UDR0 = getCharToTransmit();
		usart0Buf.setTxBufFull(0);
	}
	else usart0Buf.setTransmitActive(0);
#else  // __cplusplus
	// if chars are in a Tx buffer, transmits a char
	if (txBufStartIndex_Usart0 != txBufEndIndex_Usart0)
	{
		UDR0 = txBuf_Usart0[txBufStartIndex_Usart0++ & (TxBufLen_Usart0 - 1)];
		isTxBufFull_Usart0 = 0;  // clears 'isTxBufFull_Usart0' flag
	}
	// if all chars are transmitted
	else
	{
		isTransmitActive_Usart0 = 0;  // clears 'isTransmitActive_Usart0' flag
	}
#endif  // __cplusplus
}

ISR(USART0_RX_vect)
{
	const uint8_t ch = UDR0;  // reads a char

#if defined(__cplusplus)
	if (!usart0Buf.isRxBufFull())  // Rx buffer is not full
	{
		// TODO [check] >>
		setCharToBeReceived(ch);
	}
#else  // __cplusplus
	if (rxBufStartIndex_Usart0 + RxBufLen_Usart0 != rxBufEndIndex_Usart0)  // Rx buffer is not full
	{
		rxBuf_Usart0[rxBufEndIndex_Usart0++ & (RxBufLen_Usart0 - 1)] = ch;  // pushes a char into buffer
	}
#endif  // __cplusplus
}

ISR(USART0_UDRE_vect)
{
}

void usart0_init(const uint32_t baudrate)
{
	UDR0 = 0;

	UCSR0A = 0x00;
	//UCSR0A |= _BV(U2X0);  // double the USART0 transmission speed enable
	
	UCSR0B = 0x00;
	UCSR0B |= _BV(RXCIE0);  // USART0 RX complete interrupt enable
	UCSR0B |= _BV(TXCIE0);  // USART0 TX complete interrupt enable
	UCSR0B &= ~(_BV(UDRIE0));  // USART0 data register empty interrupt disable
	UCSR0B |= _BV(RXEN0);  // USART0 receiver enable
	UCSR0B |= _BV(TXEN0);  // USART0 transmitter enable

	UCSR0C = 0x00;
	UCSR0C &= ~(_BV(UMSEL0));  // asynchronous mode
	UCSR0C &= ~(_BV(UPM01));  // no parity
	UCSR0C &= ~(_BV(UPM00));  // no parity
	UCSR0C &= ~(_BV(USBS0));  // 1 stop bit

	// data size: 8 bits
	UCSR0B &= ~(_BV(UCSZ02));
	UCSR0C |= _BV(UCSZ01);
	UCSR0C |= _BV(UCSZ00);

	// calculate Baud rate
	//	1. asynchronous normal mode (U2Xn = 0)
	//		baudrate = Fosc / (16 * (UBRRn + 1))
	//		UBRRn = Fosc / (16 * baudrate) - 1
	//	2. asynchronous double speed mode (U2Xn = 1)
	//		baudrate = Fosc / (8 * (UBRRn + 1))
	//		UBRRn = Fosc / (8 * baudrate) - 1
	//	3. synchronous master mode
	//		baudrate = Fosc / (2 * (UBRRn + 1))
	//		UBRRn = Fosc / (2 * baudrate) - 1

	// baud rate: 57600
	//const uint32_t baudrate = 57600UL;
	const uint16_t ubrr0 = (uint16_t)(F_CPU / (16.0 * baudrate) - 1.0);
	UBRR0H = (uint8_t)((ubrr0 >> 8) & 0x00FF);
	UBRR0L = (uint8_t)(ubrr0 & 0x00FF);

#if defined(__cplusplus)
	//usart0Buf.reset();
#else  // __cplusplus
	usart0_init_buffer();
#endif  // __cplusplus
}

int8_t usart0_push_char(const uint8_t ch)
{
#if defined(__cplusplus)
	return usart0Buf.pushChar(ch);
#else  // __cplusplus
	if (!isTxBufFull_Usart0)  // transmits only if buffer is not full
	{
		if (!isTransmitActive_Usart0)  // if transmitter is not active
		{
			// transfers the first char direct to UDR0 to start transmission
			isTransmitActive_Usart0 = 1;
			UDR0 = ch;
		}
		else {
			UCSR0B &= ~(_BV(TXCIE0));  // disables USART0 TX complete interrupt after buffer is updated
			txBuf_Usart0[txBufEndIndex_Usart0++ & (TxBufLen_Usart0 - 1)] = ch;
			// puts a char to transmission buffer
			if (((txBufEndIndex_Usart0 ^ txBufStartIndex_Usart0) & (TxBufLen_Usart0 - 1)) == 0)
			{
				isTxBufFull_Usart0 = 1;
			}
			UCSR0B |= _BV(TXCIE0);  // enables USART0 TX complete interrupt after buffer is updated
		}

		return 1;
	}
	else return 0;
#endif  // __cplusplus
}

void usart0_pop_char()
{
#if defined(__cplusplus)
	usart0Buf.popChar();
#else  // __cplusplus
	if (rxBufEndIndex_Usart0 != rxBufStartIndex_Usart0)
	{
		UCSR0B &= ~(_BV(RXCIE0));  // disables USART0 RX complete interrupt after buffer is updated
		//ch = rxBuf_Usart0[++rxBufStartIndex_Usart0 & (RxBufLen_Usart0 - 1)];
		++rxBufStartIndex_Usart0;
		UCSR0B |= _BV(RXCIE0);  // enables USART0 RX complete interrupt after buffer is updated
	}
#endif  // __cplusplus
}

uint8_t usart0_top_char()
{
#if defined(__cplusplus)
	return usart0Buf.topChar();
#else  // __cplusplus
	uint8_t ch = 0xFF;

	if (rxBufEndIndex_Usart0 != rxBufStartIndex_Usart0)
	{
		UCSR0B &= ~(_BV(RXCIE0));  // disables USART0 RX complete interrupt after buffer is updated
		ch = rxBuf_Usart0[rxBufStartIndex_Usart0 & (RxBufLen_Usart0 - 1)];
		UCSR0B |= _BV(RXCIE0);  // enables USART0 RX complete interrupt after buffer is updated
	}

	return ch;
#endif  // __cplusplus
}

int8_t usart0_is_empty()
{
#if defined(__cplusplus)
	return usart0Buf.isRxBufEmpty();
#else  // __cplusplus
	return rxBufEndIndex_Usart0 == rxBufStartIndex_Usart0;
#endif  // __cplusplus
}

uint32_t usart0_get_size()
{
#if defined(__cplusplus)
	return usart0Buf.getRxBufSize();
#else  // __cplusplus
	return rxBufEndIndex_Usart0 >= rxBufStartIndex_Usart0 ? rxBufEndIndex_Usart0 - rxBufStartIndex_Usart0 : (uint32_t)((0x01 << sizeof(uint8_t)) - rxBufStartIndex_Usart0 + rxBufEndIndex_Usart0);
#endif  // __cplusplus
}

void usart0_init_buffer()
{
#if defined(__cplusplus)
	usart0Buf.reset();
#else  // __cplusplus
	// empties reception buffers
	rxBufStartIndex_Usart0 = rxBufEndIndex_Usart0 = 0;
	// empties transmission buffers
	txBufStartIndex_Usart0 = txBufEndIndex_Usart0 = 0;
	// clears 'isTxBufFull_Usart0' & 'isTransmitActive_Usart0' flags
	isTxBufFull_Usart0 = isTransmitActive_Usart0 = 0;

	for (int i = 0; i < RxBufLen_Usart0; ++i)
		rxBuf_Usart0[i] = 0;

	for (int i = 0; i < RxBufLen_Usart0; ++i)
		txBuf_Usart0[i] = 0;
#endif  // __cplusplus
}
