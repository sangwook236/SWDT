#include <boost/numeric/mtl/mtl.hpp>
#include <iostream>
#include <complex>


namespace {
namespace local {

void vector_operation()
{
	{
		// Define dense vector of doubles with 10 elements all set to 0.0.
		mtl::dense_vector<double> v(10, 0.0);

		// Set element 7 to 3.0.
		v[7] = 3.0;

		std::cout << "v is " << v << std::endl;
	}

	{
		typedef std::complex<float> complex_type;
		// Define dense vector of complex with 7 elements.
		mtl::dense_vector<complex_type, mtl::vector::parameters<mtl::tag::row_major> > v(7);

		// Set all elements to 3+2i
		v = complex_type(3.0, 2.0);
		std::cout << "v is " << v << std::endl;

		// Set all elements to 5+0i
		v = complex_type(5.0); // 5.0;
		std::cout << "v is " << v << std::endl;

		v = complex_type(6); // 6;
		std::cout << "v is " << v << std::endl;
	}

	{
		typedef std::complex<double> complex_type;
		mtl::dense_vector<complex_type> u(10), v(10);
		mtl::dense_vector<double> w(10), x(10, 4.0);

		for (size_t i = 0; i < mtl::vector::size(v); ++i)
			v[i] = complex_type(i+1, 10-i), w[i] = 2 * i + 2;

		u = v + w + x;
		std::cout << "u is " << u << std::endl;

		u -= 3 * w;
		std::cout << "u is " << u << std::endl;

		u += dot(v, w) * w + 4.0 * v + 2 * w;
		std::cout << "u is " << u << std::endl;

		std::cout << "i * w is " << complex_type(0,1) * w << std::endl;
	}
}

void vector_function()
{
	{
		typedef std::complex<double> complex_type;
		mtl::dense_vector<complex_type> v(10000), x(10, complex_type(3, 2));
		mtl::dense_vector<double> w(10000);

		for (size_t i = 0; i < mtl::vector::size(v); ++i)
			v[i] = complex_type(i+1, 10000-i), w[i] = 2 * i + 2;

		std::cout << "dot(v, w) is " << mtl::dot(v, w) << std::endl;
		std::cout << "dot<6>(v, w) is " << mtl::dot<6>(v, w) << std::endl;
		std::cout << "conj(x) is " << mtl::conj(x) << std::endl;
	}

	{
		mtl::dense_vector<double> v(100);

		for (size_t i = 0; i < mtl::vector::size(v); ++i)
			v[i] = double(i+1) * std::pow(-1.0, (int)i);

		std::cout << "max(v) is " << mtl::max(v) << std::endl;
		std::cout << "min(v) is " << mtl::min(v) << std::endl;
		std::cout << "max<6>(v) is " <<  mtl::max<6>(v) << std::endl;
	}

	{
		typedef std::complex<double> complex_type;
		mtl::dense_vector<complex_type> v(100);

		for (size_t i = 0; i < mtl::vector::size(v); ++i)
			v[i] = complex_type(i+1, 100-i);

		std::cout << "sum(v) is " << mtl::sum(v) << std::endl;
		std::cout << "product(v) is " << mtl::product(v) << std::endl;
		std::cout << "sum<6>(v) is " << mtl::sum<6>(v) << std::endl;
	}

	{
		typedef std::complex<double> complex_type;
		mtl::dense_vector<complex_type> v(10000);

		// Initialize vector
		for (size_t i = 0; i < mtl::vector::size(v); ++i)
			v[i] = complex_type(i+1, 10000-i);

		std::cout << "one_norm(v) is " << mtl::one_norm(v) << std::endl;
		std::cout << "two_norm(v) is " << mtl::two_norm(v) << std::endl;
		std::cout << "infinity_norm(v) is " << mtl::infinity_norm(v) << std::endl;

		// Unroll computation of two-norm to 6 independent statements
		std::cout << "two_norm<6>(v) is " << mtl::two_norm<6>(v) << std::endl;
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_mtl {

void vector_operation()
{
	local::vector_operation();
	local::vector_function();
}

}  // namespace my_mtl
