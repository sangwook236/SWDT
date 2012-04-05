//#include "stdafx.h"
#include <gsl/gsl_errno.h>
#include <gsl/gsl_fft_complex.h>
#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_halfcomplex.h>
#include <cstdio>
#include <cmath>


namespace {
namespace local {

#define REAL(z,i) ((z)[2*(i)])
#define IMAG(z,i) ((z)[2*(i)+1])

void fft_complex_radix2()
{
	double data[2*128];
	for (int i = 0; i < 128; ++i)
	{
		REAL(data,i) = 0.0; IMAG(data,i) = 0.0;
	}
	REAL(data,0) = 1.0;

	for (int i = 1; i <= 10; ++i)
	{
		REAL(data,i) = REAL(data,128-i) = 1.0;
	}

	for (int i = 0; i < 128; ++i)
	{
		printf("%d %e %e\n", i, REAL(data,i), IMAG(data,i));
	}
	printf ("\n");

	//
	gsl_fft_complex_radix2_forward(data, 1, 128);
	for (int i = 0; i < 128; ++i)
	{
		printf("%3d %e %e\n", i,
			REAL(data,i)/sqrt(128.0f),
			IMAG(data,i)/sqrt(128.0f));
	}
}

void fft_complex()
{
	const int n = 630;
	double data[2*n];

	for (int i = 0; i < n; ++i)
	{
		REAL(data,i) = 0.0;
		IMAG(data,i) = 0.0;
	}
	data[0] = 1.0;

	for (int i = 1; i <= 10; ++i)
	{
		REAL(data,i) = REAL(data,n-i) = 1.0;
	}
	for (int i = 0; i < n; ++i)
	{
		printf("%3d: %e %e\n", i, REAL(data,i), IMAG(data,i));
	}
	printf("\n");

	//
	gsl_fft_complex_wavetable *wavetable = gsl_fft_complex_wavetable_alloc(n);
	for (size_t i = 0; i < wavetable->nf; ++i)
	{
		printf("# factor %d: %d\n", i, wavetable->factor[i]);
	}

	gsl_fft_complex_workspace *workspace = gsl_fft_complex_workspace_alloc(n);
	gsl_fft_complex_forward(data, 1, n, wavetable, workspace);

	for (int i = 0; i < n; ++i)
	{
		printf("%3d: %e %e\n", i, REAL(data,i), IMAG(data,i));
	}

	//
	gsl_fft_complex_wavetable_free(wavetable);
	gsl_fft_complex_workspace_free(workspace);
}

void fft_real()
{
	const int n = 100;
	double data[n];

	for (int i = 0; i < n; ++i)
	{
		data[i] = 0.0;
	}
	for (int i = n / 3; i < 2 * n / 3; ++i)
	{
		data[i] = 1.0;
	}
	for (int i = 0; i < n; ++i)
	{
		printf("%3d: %e\n", i, data[i]);
	}
	printf("\n");

	//
	gsl_fft_real_wavetable *real = gsl_fft_real_wavetable_alloc(n);
	gsl_fft_real_workspace *work = gsl_fft_real_workspace_alloc(n);
	gsl_fft_real_transform(data, 1, n, real, work);
	gsl_fft_real_wavetable_free(real);
	for (int i = 11; i < n; ++i)
	{
		data[i] = 0;
	}

	gsl_fft_halfcomplex_wavetable *hc = gsl_fft_halfcomplex_wavetable_alloc(n);
	gsl_fft_halfcomplex_inverse(data, 1, n, hc, work);
	gsl_fft_halfcomplex_wavetable_free(hc);
	for (int i = 0; i < n; ++i)
	{
		printf("%3d: %e\n", i, data[i]);
	}

	//
	gsl_fft_real_workspace_free(work);
}

}  // namespace local
}  // unnamed namespace

void fft()
{
	//local::fft_complex_radix2();
	//local::fft_complex();
	local::fft_real();
}
