#include "lsd.h"
#include <iostream>

int main(void)
{
	int x, y, i, j, n;
	const int X = 128;  // x image size
	const int Y = 128;  // y image size

	// create a simple image: left half black, right half gray
	double *image = (double *)malloc(X * Y * sizeof(double));
	if (NULL == image)
	{
		std::cerr << "error: not enough memory" << std::endl;
		return EXIT_FAILURE;
	}

	for (x = 0; x < X; ++x)
		for (y = 0; y < Y; ++y)
			image[x + y * X] = (x < X / 2) ? 0.0 : 64.0;  // image(x,y)

	// LSD call
	double *out = lsd(&n, image, X, Y);

	// print output
	std::cout << n << " line segments found:" << std::endl;
	for (i = 0; i < n; ++i)
	{
		for (j = 0; j < 7; ++j)
			std::cout << out[7 * i + j] << ' ';
		std::cout << std::endl;
	}

	// free memory
	free((void *)image);
	free((void *)out);

	return EXIT_SUCCESS;
}
