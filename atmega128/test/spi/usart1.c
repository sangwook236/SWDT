#include "usart.h"
#include <avr/interrupt.h>
#include <avr/sfr_defs.h>


#if defined(__cplusplus)
class Usart1Buffer
{
public:
	Usart1Buffer()
	: rxBufLen_(64UL), txBufLen_(64UL),
	  rxBufStartIndex_(0), rxBufEndIndex_(0), txBufStartIndex_(0), txBufEndIndex_(0),
	  isTxBufFull_(0), isTransmitActive_(0)
	{
		for (int i = 0; i < rxBufLen_; ++i)
			rxBuf_[i] = 0;

		for (int i = 0; i < txBufLen_; ++i)
			txBuf_[i] = 0;
	}
	~Usart1Buffer()
	{}

private:
	Usart1Buffer(const Usart1Buffer &);  // un-implemented
	Usart1Buffer & Usart1Buffer(const Usart1Buffer &);  // un-implemented

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
				// transfers the first char direct to UDR1 to start transmission
				isTransmitActive_ = 1;
				UDR1 = ch;
			}
			else
			{
				UCSR1B &= ~(_BV(TXCIE1));  // disables USART1 TX complete interrupt after buffer is updated
				txBuf_[txBufEndIndex_++ & (txBufLen_ - 1)] = ch;
				// puts a char to transmission buffer
				if (((txBufEndIndex_ ^ txBufStartIndex_) & (txBufLen_ - 1)) == 0)
				{
					isTxBufFull_ = 1;
				}
				UCSR1B |= _BV(TXCIE1);  // enables USART1 TX complete interrupt after buffer is updated
			}

			return 1;
		}
		else return 0;
	}

	void popChar()
	{
		if (rxBufEndIndex_ != rxBufStartIndex_)
		{
			UCSR1B &= ~(_BV(RXCIE1));  // disables USART1 RX complete interrupt after buffer is updated
			//ch = rxBuf_[++rxBufStartIndex_ & (rxBufLen_ - 1)];
			++rxBufStartIndex_;
			UCSR1B |= _BV(RXCIE1);  // enables USART1 RX complete interrupt after buffer is updated
		}
	}

	uint8_t topChar() const
	{
		uint8_t ch = 0xFF;

		if (rxBufEndIndex_ != rxBufStartIndex_)
		{
			UCSR1B &= ~(_BV(RXCIE1));  // disables USART1 RX complete interrupt after buffer is updated
			ch = rxBuf_[rxBufStartIndex_ & (rxBufLen_ - 1)];
			UCSR1B |= _BV(RXCIE1);  // enables USART1 RX complete interrupt after buffer is updated
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

static volatile Usart1Buffer usart1Buf;
#else  // __cplusplus
#define RxBufLen_Usart1 64  // a size of USART1 reception buffer
static volatile uint8_t rxBufStartIndex_Usart1, rxBufEndIndex_Usart1;  // start & end indexes of USART1 reception buffer
static volatile uint8_t rxBuf_Usart1[RxBufLen_Usart1];  // a storage for USART1 reception buffer

#define TxBufLen_Usart1 64  // a size of USART1 transmission buffer
static volatile uint8_t txBufStartIndex_Usart1, txBufEndIndex_Usart1;  // start & end indexes of USART1 transmission buffer
static volatile uint8_t txBuf_Usart1[TxBufLen_Usart1];  // a storage for USART1 transmission buffer

static volatile int8_t isTxBufFull_Usart1;  // flag: marks a USART1 transmission buffer full
static volatile int8_t isTransmitActive_Usart1;  // flag: marks a USART1 transmitter active
#endif  // __cplusplus

ISR(USART1_TX_vect)
{
#if defined(__cplusplus)
	// if chars are in a Tx buffer, transmits a char
	if (!usart1Buf.isTxBufEmpty())
	{
		// TODO [check] >>
		UDR1 = getCharToTransmit();
		usart1Buf.setTxBufFull(0);
	}
	else usart1Buf.setTransmitActive(0);
#else  // __cplusplus
	// if chars are in a Tx buffer, transmits a char
	if (txBufStartIndex_Usart1 != txBufEndIndex_Usart1)
	{
		UDR1 = txBuf_Usart1[txBufStartIndex_Usart1++ & (TxBufLen_Usart1 - 1)];
		isTxBufFull_Usart1 = 0;  // clears 'isTxBufFull_Usart1' flag
	}
	// if all chars are transmitted
	else
	{
		isTransmitActive_Usart1 = 0;  // clears 'isTransmitActive_Usart1' flag
	}
#endif  // __cplusplus
}

ISR(USART1_RX_vect)
{
	const uint8_t ch = UDR1;  // reads a char

#if defined(__cplusplus)
	if (!usart1Buf.isRxBufFull())  // Rx buffer is not full
	{
		// TODO [check] >>
		setCharToBeReceived(ch);
	}
#else  // __cplusplus
	if (rxBufStartIndex_Usart1 + RxBufLen_Usart1 != rxBufEndIndex_Usart1)  // Rx buffer is not full
	{
		rxBuf_Usart1[rxBufEndIndex_Usart1++ & (RxBufLen_Usart1 - 1)] = ch;  // pushes a char into buffer
	}
#endif  // __cplusplus
}

ISR(USART1_UDRE_vect)
{
}

void usart1_init()
{
	UDR1 = 0;

	UCSR1A = 0x00;
	//UCSR1A |= _BV(U2X1);  // double the USART1 transmission speed enable
	
	UCSR1B = 0x00;
	UCSR1B |= _BV(RXCIE1);  // USART1 RX complete interrupt enable
	UCSR1B |= _BV(TXCIE1);  // USART1 TX complete interrupt enable
	UCSR1B &= ~(_BV(UDRIE1));  // USART1 data register empty interrupt disable
	UCSR1B |= _BV(RXEN1);  // USART1 receiver enable
	UCSR1B |= _BV(TXEN1);  // USART1 transmitter enable

	UCSR1C = 0x00;
	UCSR1C &= ~(_BV(UMSEL1));  // asynchronous mode
	UCSR1C &= ~(_BV(UPM11));  // no parity
	UCSR1C &= ~(_BV(UPM10));  // no parity
	UCSR1C &= ~(_BV(USBS1));  // 1 stop bit

	 // data size: 8 bits
	UCSR1B &= ~(_BV(UCSZ12));
	UCSR1C |= _BV(UCSZ11);
	UCSR1C |= _BV(UCSZ10);

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
	const uint32_t baudrate = 57600UL;
	const uint16_t ubrr1 = (uint16_t)(F_CPU / (16.0 * baudrate) - 1.0);
	UBRR1H = (uint8_t)((ubrr1 & 0xFF00) >> 8);
	UBRR1L = (uint8_t)(ubrr1 & 0x00FF);

#if defined(__cplusplus)
	//usart1Buf.reset();
#else  // __cplusplus
	usart1_init_buffer();
#endif  // __cplusplus
}

int8_t usart1_push_char(const uint8_t ch)
{
#if defined(__cplusplus)
	return usart1Buf.pushChar(ch);
#else  // __cplusplus
	if (!isTxBufFull_Usart1)  // transmits only if buffer is not full
	{
		if (!isTransmitActive_Usart1)  // if transmitter is not active
		{
			// transfers the first char direct to UDR1 to start transmission
			isTransmitActive_Usart1 = 1;
			UDR1 = ch;
		}
		else {
			UCSR1B &= ~(_BV(TXCIE1));  // disables USART1 TX complete interrupt after buffer is updated
			txBuf_Usart1[txBufEndIndex_Usart1++ & (TxBufLen_Usart1 - 1)] = ch;
			// puts a char to transmission buffer
			if (((txBufEndIndex_Usart1 ^ txBufStartIndex_Usart1) & (TxBufLen_Usart1 - 1)) == 0)
			{
				isTxBufFull_Usart1 = 1;
			}
			UCSR1B |= _BV(TXCIE1);  // enables USART1 TX complete interrupt after buffer is updated
		}

		return 1;
	}
	else return 0;
#endif  // __cplusplus
}

void usart1_pop_char()
{
#if defined(__cplusplus)
	usart1Buf.popChar();
#else  // __cplusplus
	if (rxBufEndIndex_Usart1 != rxBufStartIndex_Usart1)
	{
		UCSR1B &= ~(_BV(RXCIE1));  // disables USART1 RX complete interrupt after buffer is updated
		//ch = rxBuf_Usart1[++rxBufStartIndex_Usart1 & (RxBufLen_Usart1 - 1)];
		++rxBufStartIndex_Usart1;
		UCSR1B |= _BV(RXCIE1);  // enables USART1 RX complete interrupt after buffer is updated
	}
#endif  // __cplusplus
}

uint8_t usart1_top_char()
{
#if defined(__cplusplus)
	return usart1Buf.topChar();
#else  // __cplusplus
	uint8_t ch = 0xFF;

	if (rxBufEndIndex_Usart1 != rxBufStartIndex_Usart1)
	{
		UCSR1B &= ~(_BV(RXCIE1));  // disables USART1 RX complete interrupt after buffer is updated
		ch = rxBuf_Usart1[rxBufStartIndex_Usart1 & (RxBufLen_Usart1 - 1)];
		UCSR1B |= _BV(RXCIE1);  // enables USART1 RX complete interrupt after buffer is updated
	}

	return ch;
#endif  // __cplusplus
}

int8_t usart1_is_empty()
{
#if defined(__cplusplus)
	return usart1Buf.isRxBufEmpty();
#else  // __cplusplus
	return rxBufEndIndex_Usart1 == rxBufStartIndex_Usart1;
#endif  // __cplusplus
}

uint32_t usart1_get_size()
{
#if defined(__cplusplus)
	return usart1Buf.getRxBufSize();
#else  // __cplusplus
	return rxBufEndIndex_Usart1 >= rxBufStartIndex_Usart1 ? rxBufEndIndex_Usart1 - rxBufStartIndex_Usart1 : (uint32_t)((0x01 << sizeof(uint8_t)) - rxBufStartIndex_Usart1 + rxBufEndIndex_Usart1);
#endif  // __cplusplus
}

void usart1_init_buffer()
{
#if defined(__cplusplus)
	usart1Buf.reset();
#else  // __cplusplus
	// empties reception buffers
	rxBufStartIndex_Usart1 = rxBufEndIndex_Usart1 = 0;
	// empties transmission buffers
	txBufStartIndex_Usart1 = txBufEndIndex_Usart1 = 0;
	// clears 'isTxBufFull_Usart1' & 'isTransmitActive_Usart1' flags
	isTxBufFull_Usart1 = isTransmitActive_Usart1 = 0;

	for (int i = 0; i < RxBufLen_Usart1; ++i)
		rxBuf_Usart1[i] = 0;

	for (int i = 0; i < RxBufLen_Usart1; ++i)
		txBuf_Usart1[i] = 0;
#endif  // __cplusplus
}
