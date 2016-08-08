#pragma once

#if !defined(__SHARED_LIB__TRIGONOMETRIC_H__)
#define __SHARED_LIB__TRIGONOMETRIC_H__ 1


#if defined(SHAREDLIB_EXPORTS)
#define SHAREDLIB_API __declspec(dllexport)
#else
#define SHAREDLIB_API __declspec(dllimport)
#endif


SHAREDLIB_API extern const double E;
SHAREDLIB_API double SQRT(const double val);

class SHAREDLIB_API Trigonometric
{
public:
	Trigonometric(const double val)
	: val_(val)
	{}

public:
	static double sin(const double val);

public:
	static const double PI;

public:
	double cos() const;

	void tan();

	double getValue() { return val_; }

private:
	double val_;
};

#endif  // __SHARED_LIB__TRIGONOMETRIC_H__
