// microsoft_specific.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>

#pragma pack(8)

struct __declspec(novtable) X
{
public:
	virtual void vf()
	{
		std::cout << "In X" << std::endl;
	}

private:
	int i_;
	double d_;
};

struct Y : public X
{
public:
	void vf()
	{
		std::cout << "In Y" << std::endl;
	}

private:
	int i_;
	double d_;
};

struct Z
{
public:
	void nvf()
	{
		std::cout << "In Z" << std::endl;
	}

private:
	int i_;
	double d_;
};

class V
{
public:
	void nvf()
	{
		std::cout << "In V" << std::endl;
	}

public:
	int i_;
	double d_;
};

class W
{
public:
	virtual void vf()
	{
		std::cout << "In W" << std::endl;
	}

public:
	int i_;
	long l_;
	float f_;
	double d_;
	char c_;
};

int main(int argc, char **argv)
{
	try
	{
		X *pX = new X();
		//pX->vf();  // Runtime-error.

		Y *pY = new Y();
		pY->vf();

		Z *pZ = new Z();
		pZ->nvf();

		V *pV = new V();
		pV->nvf();

		W *pW = new W();
		pW->vf();

		char * const pp1 = (char *)pV;
		char * const pp2 = (char *)(&pV->i_);

		char * const p1 = (char *)pW;
		char * const p2 = (char *)(&pW->i_);
		char * const p3 = (char *)(&pW->l_);
		char * const p4 = (char *)(&pW->f_);
		char * const p5 = (char *)(&pW->d_);
		char * const p6 = (char *)(&pW->c_);
		const size_t s1 = sizeof(pW);
		const size_t s2 = sizeof(&pW->i_);
		const size_t s3 = sizeof(&pW->d_);

		delete pX;
		delete pY;
		delete pZ;
		delete pV;
		delete pW;
	}
	catch (const std::exception &e)
	{
		std::cout << "std::exception caught !!!: " << e.what() << std::endl;
	}
	catch (...)
	{
		std::cout << "Unknown exception caught !!!" << std::endl;
	}

	std::cout << "Press any key to exit ..." << std::endl;
	std::cin.get();

    return 0;
}
