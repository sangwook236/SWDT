#include <cppad/cppad.hpp>
#include <iostream>
#include <vector>
#include <limits>
#include <cmath>


namespace {
namespace local {

// REF [site] >> http://www.coin-or.org/CppAD/Doc/add.cpp.htm
void forward_and_reverse_derivative_example()
{
	// Domain space vector.
	const size_t n = 1;
	const double x0 = 0.5;
	std::vector<CppAD::AD<double>> x(n);
	x[0] = x0;

	// Declare independent variables and start tape recording.
	CppAD::Independent(x);

	// Some binary addition operations.
	CppAD::AD<double> a = x[0] + 1.;  // CppAD::AD<double> + double.
	CppAD::AD<double> b = a + 2;  // CppAD::AD<double> + int.
	CppAD::AD<double> c = 3. + b;  // double + CppAD::AD<double>.
	CppAD::AD<double> d = 4 + c;  // int + CppAD::AD<double>.

	// Range space vector.
	const size_t m = 1;
	std::vector<CppAD::AD<double>> y(m);
	// y = 2 * x + 10.
	y[0] = d + x[0];  // CppAD::AD<double> + CppAD::AD<double>

	// Create f: x -> y and stop tape recording.
	CppAD::ADFun<double> f(x, y);

	//std::cout << "y = " << y[0] << ", the true value of y = " << (2.0 * x0 + 10.0) << std::endl;

	// Forward computation of partials w.r.t. x[0].
	std::vector<double> dx(n);
	dx[0] = 1.0;
	const std::vector<double> &dy = f.Forward(1, dx);
	std::cout << "Forward derivative dy of y = " << dy[0] << ", the true value of dy = 2.0" << std::endl;

	// Reverse computation of derivative of y[0].
	std::vector<double> w(m);
	w[0] = 1.0;
	const std::vector<double> &dw = f.Reverse(1, w);
	std::cout << "Reverse derivative dy of y = " << dw[0] << ", the true value of dy = 2.0" << std::endl;
}

// REF [file] >> ${COIN-OR_HOME}/cppad/example/erf.cpp
void erf_example()
{
	//const double eps = 10. * CppAD::numeric_limits<double>::epsilon();

	// Domain space vector.
	const size_t n = 1;
	const double x0 = 0.5;
	std::vector<CppAD::AD<double>> ax(n);
	ax[0] = x0;

	// Declare independent variables and start tape recording.
	CppAD::Independent(ax);

	// Range space vector.
	const size_t m = 1;
	std::vector<CppAD::AD<double>> ay(m);
	ay[0] = CppAD::erf(ax[0]);

	// Create f: x -> y and stop tape recording.
	CppAD::ADFun<double> f(ax, ay);

	// Value of derivative of erf at x0.
	const double pi = 4.0 * std::atan(1.0);
	const double factor = 2.0 / std::sqrt(pi);
	const double derf_dx = factor * std::exp(-x0 * x0);
	//const double erf_x0 = 0.52050;

	// Forward computation of first partial w.r.t. x[0].
	std::vector<double> dx(n);
	dx[0] = 1.0;
	const std::vector<double> &dy = f.Forward(1, dx);
	std::cout << "Forward derivative dy of y = " << dy[0] << ", the true value of dy = " << derf_dx << std::endl;

	// Reverse computation of derivative of y[0].
	std::vector<double> w(m);
	w[0] = 1.0;
	const std::vector<double> &dw = f.Reverse(1, w);
	std::cout << "Reverse derivative dy of y = " << dw[0] << ", the true value of dy = " << derf_dx << std::endl;
}

// REF [function] >> CppAD::Poly in ${COIN-OR_HOME}/cppad/poly.hpp
// Define y(x) = Poly(a, x).
template<typename T>
struct Polynomial
{
public:
	typedef T value_type;

public:
	Polynomial(const std::vector<T> &coeffs)
	: coeffs_(coeffs)
	{}

	template<typename Type>
	Type operator()(const Type &x) const
	{
		const size_t k = coeffs_.size();
		Type y = 0.;  // Initialize summation.
		Type x_i = 1.;  // Initialize x^i.
		for (size_t i = 0; i < k; i++)
		{
			y += coeffs_[i] * x_i;  // y = y + a_i * x^i.
			x_i *= x;  // x_i = x_i * x.
		}
		return y;
	}

private:
	const std::vector<T> &coeffs_;
};

// REF [site] >> http://www.coin-or.org/CppAD/Doc/get_started.cpp.xml
// REF [file] >> ${COIN-OR_HOME}/cppad/example/poly.cpp
template<class Function, typename Type = Function::value_type>
Type function_derivative_example(Function func, const Type &val)
{
	// Domain space vector.
	const size_t n = 1;  // Number of domain space variables.
	std::vector<CppAD::AD<Type>> X(n);  // Vector of domain space variables.
	X[0] = val;  // Value corresponding to operation sequence.

	// Declare independent variables and start recording operation sequence.
	CppAD::Independent(X);

	// Range space vector.
	const size_t m = 1;  // Number of ranges space variables.
	std::vector<CppAD::AD<Type>> Y(m);  // Vector of ranges space variables.
	Y[0] = func(X[0]);  // Value during recording of operations.

	// Store operation sequence in f: X -> Y and stop recording.
	CppAD::ADFun<Type> f(X, Y);

	// Compute derivative using operation sequence stored in f.
	std::vector<Type> x(n);  // Domain space vector.
	x[0] = val;  // Argument value for derivative.
	const std::vector<Type> &jac = f.Jacobian(x);  // Jacobian for operation sequence.

	return jac[0];
}

void function_jacobian_example()
{
	// Vector of polynomial coefficients.
	const size_t k = 5;  // Number of polynomial coefficients.
	std::vector<double> coeffs(k);  // Vector of polynomial coefficients.
	for (size_t i = 0; i < k; ++i)
		coeffs[i] = 1.0;  // Value of polynomial coefficients.

	const double jacobian = function_derivative_example(Polynomial<double>(coeffs), 3.0);

	// Print the results.
	std::cout << "f'(3) computed by CppAD = " << jacobian << std::endl;
	std::cout << "The true value of f'(3) = 142.0" << std::endl;
}

}  // namespace local
}  // unnamed namespace

namespace my_cppad {

void eigen();

}  // namespace my_cppad

int cppad_main(int argc, char *argv[])
{
	// Forward and reverse derivatives.
	//local::forward_and_reverse_derivative_example();
	//local::erf_example();

	// Derivative of functions such as polynomials.
	local::function_jacobian_example();

	// Support Eigen.
	//my_cppad::eigen();

	return 0;
}

