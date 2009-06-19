#include <avr/interrupt.h>
#include <avr/sfr_defs.h>


uint8_t hex2ascii(const uint8_t hex);
uint8_t ascii2hex(const uint8_t ascii);

#if defined(__cplusplus)
class UsartBuffer;

static volatile UsartBuffer usart0Buf;
#else  // __cplusplus
#define RxBufLen_Usart0 64  // a size of USART0 reception buffer
static volatile uint8_t rxBufStartIndex_Usart0, rxBufEndIndex_Usart0;  // start & end indexes of USART0 reception buffer
static volatile uint8_t rxBuf_Usart0[RxBufLen_Usart0];  // a storage for USART0 reception buffer

#define TxBufLen_Usart0 64  // a size of USART0 transmission buffer
static volatile uint8_t txBufStartIndex_Usart0, txBufEndIndex_Usart0;  // start & end indexes of USART0 transmission buffer
static volatile uint8_t txBuf_Usart0[TxBufLen_Usart0];  // a storage for USART0 transmission buffer

static volatile int8_t isTxBufFull_Usart0;  // flag: marks a USART0 transmission buffer full
static volatile int8_t isTransmitActive_Usart0;  // flag: marks a USART0 transmitter active

int8_t pushChar_Usart0(const uint8_t ch);
void popChar_Usart0();
uint8_t topChar_Usart0();
int8_t isEmpty_Usart0();
uint32_t getSize_Usart0();

void initUsartBuf_usart0();
#endif  // __cplusplus

ISR(USART0_TX_vect)
{
#if defined(__cplusplus)
	// if chars are in a Tx buffer, transmits a char
	if (!usart0Buf.isTxBufEmpty()) {
		// need to add

		usart0Buf.setTxBufFull(0);
	}
	else usart0Buf.setTransmitActive(0);
#else  // __cplusplus
	// if chars are in a Tx buffer, transmits a char
	if (txBufStartIndex_Usart0 != txBufEndIndex_Usart0) {
		UDR0 = txBuf_Usart0[txBufStartIndex_Usart0++ & (TxBufLen_Usart0 - 1)];
		isTxBufFull_Usart0 = 0;  // clears 'isTxBufFull_Usart0' flag
	}
	// if all chars are transmitted
	else {
		isTransmitActive_Usart0 = 0;  // clears 'isTransmitActive_Usart0' flag
	}
#endif  // __cplusplus
}

ISR(USART0_RX_vect)
{
	const uint8_t ch = UDR0;  // reads a char

#if defined(__cplusplus)
	if (!usart0Buf.isRxBufFull()) {  // Rx buffer is not full
		// need to add
	}
#else  // __cplusplus
	if (rxBufStartIndex_Usart0 + RxBufLen_Usart0 != rxBufEndIndex_Usart0) {  // Rx buffer is not full
		rxBuf_Usart0[rxBufEndIndex_Usart0++ & (RxBufLen_Usart0 - 1)] = ch;  // pushes a char into buffer
	}
#endif  // __cplusplus
}

ISR(USART0_UDRE_vect)
{
}

void initUsart()
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
	const uint32_t baudrate = 57600UL;
	const uint16_t ubrr0 = (uint16_t)(F_CPU / (16.0 * baudrate) - 1.0);
	UBRR0H = (uint8_t)((ubrr0 & 0xFF00) >> 8);
	UBRR0L = (uint8_t)(ubrr0 & 0x00FF);

#if defined(__cplusplus)
	//usart0Buf.reset();
#else  // __cplusplus
	initUsartBuf_usart0();
#endif  // __cplusplus
}

uint8_t hex2ascii(const uint8_t hex)
{
	if (0x00 <= (int8_t)hex && (int8_t)hex <= 0x09)
		return hex + '0';
	//else if (0x0a <= hex && hex <= 0x0f)
	else if (0x0A <= hex && hex <= 0x0F)
		//return hex - 0x0A + (doesConvToUpperCase ? 'A' : 'a');
		return hex - 0x0A + 'A';
	else return (uint8_t)-1;
}

uint8_t ascii2hex(const uint8_t ascii)
{
	if ('0' <= ascii && ascii <= '9')
		return ascii - '0';
	else if ('a' <= ascii && ascii <= 'f')
		return ascii - 'a' + 10;
	else if ('A' <= ascii && ascii <= 'F')
		return ascii - 'A' + 10;
	else return (uint8_t)-1;
}

#if defined(__cplusplus)
class UsartBuffer {
public:
	const uint32_t RxBufLen;
	const uint32_t TxBufLen;

public:
	UsartBuffer()
	: RxBufLen(64UL), TxBufLen(64UL),
		rxBufStartIndex_(0), rxBufEndIndex_(0),
		txBufStartIndex_(0), txBufEndIndex_(0),
		isTxBufFull_(0), isTransmitActive_(0)
	{
		for (int i = 0; i < RxBufLen_Usart0; ++i)
			rxBuf_[i] = 0;

		for (int i = 0; i < RxBufLen_Usart0; ++i)
			txBuf_[i] = 0;
	}
	~UsartBuffer()
	{}

private:
	UsartBuffer(const UsartBuffer& rhs);  // un-implemented
	UsartBuffer & UsartBuffer(const UsartBuffer& rhs);  // un-implemented

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
		if (!isTxBufFull_Usart0) {  // transmits only if buffer is not full
			if (!isTransmitActive_Usart0) {  // if transmitter is not active
				// transfers the first char direct to UDR0 to start transmission
				isTransmitActive_Usart0 = 1;
				UDR0 = ch;
			}
			else {
				UCSR0B &= ~(_BV(TXCIE0));  // disables USART0 TX complete interrupt after buffer is updated
				txBuf_Usart0[txBufEndIndex_Usart0++ & (TxBufLen_Usart0 - 1)] = ch;
				// puts a char to transmission buffer
				if (((txBufEndIndex_Usart0 ^ txBufStartIndex_Usart0) & (TxBufLen_Usart0 - 1)) == 0) {
					isTxBufFull_Usart0 = 1;
				}
				UCSR0B |= _BV(TXCIE0);  // enables USART0 TX complete interrupt after buffer is updated
			}

			return 1;
		}
		else return 0;
	}

	void popChar()
	{
		if (rxBufEndIndex_ != rxBufStartIndex_) {
			UCSR0B &= ~(_BV(RXCIE0));  // disables USART0 RX complete interrupt after buffer is updated
			//ch = rxBuf_[++rxBufStartIndex_ & (RxBufLen - 1)];
			++rxBufStartIndex_;
			UCSR0B |= _BV(RXCIE0);  // enables USART0 RX complete interrupt after buffer is updated
		}
	}

	uint8_t topChar() const
	{
		uint8_t ch = 0xFF;

		if (rxBufEndIndex_ != rxBufStartIndex_) {
			UCSR0B &= ~(_BV(RXCIE0));  // disables USART0 RX complete interrupt after buffer is updated
			ch = rxBuf_[rxBufStartIndex_ & (RxBufLen - 1)];
			UCSR0B |= _BV(RXCIE0);  // enables USART0 RX complete interrupt after buffer is updated
		}

		return ch;
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

private:
	// start & end indexes of USART0 reception buffer
	uint8_t rxBufStartIndex_, rxBufEndIndex_;
	// a storage for USART0 reception buffer
	uint8_t rxBuf_[RxBufLen];

	// start & end indexes of USART0 transmission buffer
	uint8_t txBufStartIndex_, txBufEndIndex_;
	// a storage for USART0 transmission buffer
	uint8_t txBuf_[TxBufLen];

	// marks a USART0 transmission buffer flag full
	int8_t isTxBufFull_;
	// marks a USART0 transmitter flag active
	int8_t isTransmitActive_;
};
#else  // __cplusplus
int8_t pushChar_Usart0(const uint8_t ch)
{
	if (!isTxBufFull_Usart0) {  // transmits only if buffer is not full
		if (!isTransmitActive_Usart0) {  // if transmitter is not active
			// transfers the first char direct to UDR0 to start transmission
			isTransmitActive_Usart0 = 1;
			UDR0 = ch;
		}
		else {
			UCSR0B &= ~(_BV(TXCIE0));  // disables USART0 TX complete interrupt after buffer is updated
			txBuf_Usart0[txBufEndIndex_Usart0++ & (TxBufLen_Usart0 - 1)] = ch;
			// puts a char to transmission buffer
			if (((txBufEndIndex_Usart0 ^ txBufStartIndex_Usart0) & (TxBufLen_Usart0 - 1)) == 0) {
				isTxBufFull_Usart0 = 1;
			}
			UCSR0B |= _BV(TXCIE0);  // enables USART0 TX complete interrupt after buffer is updated
		}

		return 1;
	}
	else return 0;
}

void popChar_Usart0()
{
	if (rxBufEndIndex_Usart0 != rxBufStartIndex_Usart0) {
		UCSR0B &= ~(_BV(RXCIE0));  // disables USART0 RX complete interrupt after buffer is updated
		//ch = rxBuf_Usart0[++rxBufStartIndex_Usart0 & (RxBufLen_Usart0 - 1)];
		++rxBufStartIndex_Usart0;
		UCSR0B |= _BV(RXCIE0);  // enables USART0 RX complete interrupt after buffer is updated
	}
}

uint8_t topChar_Usart0()
{
	uint8_t ch = 0xFF;

	if (rxBufEndIndex_Usart0 != rxBufStartIndex_Usart0) {
		UCSR0B &= ~(_BV(RXCIE0));  // disables USART0 RX complete interrupt after buffer is updated
		ch = rxBuf_Usart0[rxBufStartIndex_Usart0 & (RxBufLen_Usart0 - 1)];
		UCSR0B |= _BV(RXCIE0);  // enables USART0 RX complete interrupt after buffer is updated
	}

	return ch;
}

int8_t isEmpty_Usart0()
{  return rxBufEndIndex_Usart0 == rxBufStartIndex_Usart0;  }

uint32_t getSize_Usart0()
{
	return rxBufEndIndex_Usart0 >= rxBufStartIndex_Usart0 ? rxBufEndIndex_Usart0 - rxBufStartIndex_Usart0 : (uint32_t)((0x01 << sizeof(uint8_t)) - rxBufStartIndex_Usart0 + rxBufEndIndex_Usart0);
}

void initUsartBuf_usart0()
{
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
}
#endif  // __cplusplus
