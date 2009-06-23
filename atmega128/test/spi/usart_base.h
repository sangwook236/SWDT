#if !defined(__SWL_AVR__USART_BASE_H_)
#define __SWL_AVR__USART_BASE_H_ 1


#if defined(__cplusplus)
class UsartBuffer
{
public:
	UsartBuffer()
	: RxBufLen(64UL), TxBufLen(64UL),
	  rxBufStartIndex_(0), rxBufEndIndex_(0), txBufStartIndex_(0), txBufEndIndex_(0),
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
		if (!isTxBufFull_Usart0)  // transmits only if buffer is not full
		{
			if (!isTransmitActive_Usart0)  // if transmitter is not active
			{
				// transfers the first char direct to UDR0 to start transmission
				isTransmitActive_Usart0 = 1;
				UDR0 = ch;
			}
			else
			{
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
	}

	void popChar()
	{
		if (rxBufEndIndex_ != rxBufStartIndex_)
		{
			UCSR0B &= ~(_BV(RXCIE0));  // disables USART0 RX complete interrupt after buffer is updated
			//ch = rxBuf_[++rxBufStartIndex_ & (RxBufLen - 1)];
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

public:
	const uint32_t RxBufLen;
	const uint32_t TxBufLen;

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
#endif  // __cplusplus


#endif  // __SWL_AVR__USART_BASE_H_
