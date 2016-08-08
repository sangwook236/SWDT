#pragma once

#if !defined(__STATIC_LIB__ARITHMETIC_H__)
#define __STATIC_LIB__ARITHMETIC_H__ 1


extern const int TWO;
int DOUBLE(const int val);

class Arithmetic
{
public:
	struct InnerStruct
	{
	public:
		InnerStruct(const int val)
		: val_(val)
		{}

	public:
		int add(const int lhs) const;

	private:
		int val_;
	};

	class InnerClass
	{
	public:
		InnerClass(const int val)
		: val_(val)
		{}

	public:
		int subtract(const int lhs) const;

	private:
		int val_;
	};

public:
	Arithmetic(const int lhs)
	: lhs_(lhs)
	{}

public:
	static int add(const int lhs, const int rhs);
	static int subtract(const int lhs, const int rhs);

public:
	static const int ONE;

public:
	int add(const int rhs) const;
	int subtract(const int rhs) const;

	void multiply(const int rhs);

	int getValue() { return lhs_; }

private:
	int lhs_;
};

#endif  // __STATIC_LIB__ARITHMETIC_H__
