#include "stdafx.h"
#include <atlas_enum.h>
#include <cblas.h>
#include <iostream>


void cblas()
{
	const double m[] = {
		3, 1, 3,
		1, 5, 9,
		2, 6, 5
	};
	const double x[] = { -1, -1, 1 };
	double y[] = { 0, 0, 0 };

	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
			std::cout << m[i*3 + j] << " ";
		std::cout << std::endl;
	}

	cblas_dgemv(CblasRowMajor, CblasNoTrans, 3, 3, 1.0, m, 3, x, 1, 0.0, y, 1);

	for (int i = 0; i < 3; ++i)
		std::cout << y[i] << std::endl;
}
