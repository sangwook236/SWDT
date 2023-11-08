//#include "stdafx.h"
//#define EIGEN2_SUPPORT 1
#include <cmath>
#include <string>
#include <iostream>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>


namespace {
namespace local {

auto my_transpose(const Eigen::Tensor<float, 2> &tensor)
{
	Eigen::array<int, 2> dims({1, 0});
	return tensor.shuffle(dims);
}

struct MyMaxReducer
{
	void reduce(float t, float *accum) const
	{
		*accum = *accum >= t ? *accum : t;
	}
	float initialize() const
	{
		return -std::numeric_limits<float>::max();
	}
	float finalize(float accum) const
	{
		return accum;
	}
};

auto my_softmax(const Eigen::Tensor<float, 3> &z)
{
	auto dimensions = z.dimensions();

	int batches = dimensions.at(0);
	int instances_per_batch = dimensions.at(1);
	int instance_length = dimensions.at(2);

	Eigen::array<int, 1> depth_dim({2});
	auto z_max = z.maximum(depth_dim);

	Eigen::array<int, 3> reshape_dim({batches, instances_per_batch, 1});
	auto max_reshaped = z_max.reshape(reshape_dim);

	Eigen::array<int, 3> bcast({1, 1, instance_length});
	auto max_values = max_reshaped.broadcast(bcast);

	auto diff = z - max_values;

	auto expo = diff.exp();
	auto expo_sums = expo.sum(depth_dim);
	auto sums_reshaped = expo_sums.reshape(reshape_dim);
	auto sums = sums_reshaped.broadcast(bcast);
	auto result = expo / sums;

	return result;
}

// REF [site] >> https://eigen.tuxfamily.org/dox-devel/unsupported/eigen_tensors.html
void simple_example()
{
	// Tensor classes.

	{
		// Create a tensor of rank 3 of sizes 2, 3, 4.
		// This tensor owns memory to hold 24 floating point values (24 = 2 x 3 x 4).
		Eigen::Tensor<float, 3> t_3d(2, 3, 4);
		t_3d.setValues({{{1, 2, 3, 4}, {5, 6, 7, 8}}});
		//t_3d.setRandom();
		std::cout << "t_3d:\n" << t_3d << std::endl;

		// Resize t_3d by assigning a tensor of different sizes, but same rank.
		t_3d = Eigen::Tensor<float, 3>(3, 4, 3);

		// Create a tensor of strings of rank 2 with sizes 5, 7.
		Eigen::Tensor<std::string, 2> t_2d(5, 7);
	}

	{
		// Fixed sized tensors can provide very fast computations because all their dimensions are known by the compiler.
		// FixedSize tensors are not resizable.

		// Create a 4 x 3 tensor of floats.
		Eigen::TensorFixedSize<float, Eigen::Sizes<4, 3>> t_4x3;
	}

	{
		// A TensorMap is not resizable because it does not own the memory where its data are stored.

		// Map a tensor of ints on top of stack-allocated storage.
		int storage[128] = {0,};  // 2 x 4 x 2 x 8 = 128.
		Eigen::TensorMap<Eigen::Tensor<int, 4>> t_4d(storage, 2, 4, 2, 8);

		// The same storage can be viewed as a different tensor.
		// You can also pass the sizes as an array.
		Eigen::TensorMap<Eigen::Tensor<int, 2>> t_2d(storage, 16, 8);

		// You can also map fixed-size tensors.
		// Here we get a 1d view of the 2d fixed-size tensor.
		Eigen::TensorFixedSize<float, Eigen::Sizes<4, 3>> t_4x3;
		Eigen::TensorMap<Eigen::Tensor<float, 1>> tm_12(t_4x3.data(), 12);
	}

	{
		// Accessing tensor elements.

		// Set the value of the element at position (0, 1, 0).
		Eigen::Tensor<float, 3> t_3d(2, 3, 4);
		t_3d(0, 1, 0) = 12.0f;

		// Initialize all elements to random values.
		for (int i = 0; i < 2; ++i)
			for (int j = 0; j < 3; ++j)
				for (int k = 0; k < 4; ++k)
					t_3d(i, j, k) = float(i + j + k);

		// Print elements of a tensor.
		for (int i = 0; i < 2; ++i)
			std::cout << t_3d(i, 0, 0) << ", ";
		std::cout << std::endl;
	}

	{
		// Tensor layout.
		// The tensor library supports 2 layouts: ColMajor (the default) and RowMajor.

		Eigen::Tensor<float, 3, Eigen::ColMajor> col_major0;  // Equivalent to Eigen::Tensor<float, 3>.
		float data[30] = {0.0f,};  // 3 x 2 x 5 = 30.
		Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor> > row_major0(data, 3, 2, 5);

		Eigen::Tensor<float, 2, Eigen::ColMajor> col_major(2, 4);
		Eigen::Tensor<float, 2, Eigen::RowMajor> row_major(2, 4);

		Eigen::Tensor<float, 2> col_major_result = col_major;  // Ok, layouts match.
		//Eigen::Tensor<float, 2> col_major_result = row_major;  // Will not compile.

		// Simple layout swap.
		col_major_result = row_major.swap_layout();
		eigen_assert(col_major_result.dimension(0) == 4);
		eigen_assert(col_major_result.dimension(1) == 2);

		// Swap the layout and preserve the order of the dimensions.
		std::array<int, 2> shuffle({1, 0});
		col_major_result = row_major.swap_layout().shuffle(shuffle);
		eigen_assert(col_major_result.dimension(0) == 2);
		eigen_assert(col_major_result.dimension(1) == 4);
	}

	//--------------------
	// Tensor operations.

	{
		Eigen::Tensor<float, 3> t1(2, 3, 4);
		t1.setConstant(3);
		Eigen::Tensor<float, 3> t2(2, 3, 4);
		t2.setConstant(7);

		// Set t3 to the element wise sum of t1 and t2.
		Eigen::Tensor<float, 3> t3 = t1 + t2;
		// t4 is actually the tree of tensor operators that will compute the addition of t1 and t2.
		// In fact, t4 is not a tensor and you cannot get the values of its elements.
		auto t4 = t1 + t2;

		std::cout << t3(0, 0, 0) << std::endl;  // OK prints the value of t1(0, 0, 0) + t2(0, 0, 0).
		//std::cout << t4(0, 0, 0) << std::endl;  // Compilation error!

		Eigen::Tensor<float, 3> result1 = t4;  // Could also be: result1(t4).
		std::cout << result1(0, 0, 0) << std::endl;

		//-----
		// Tensor Operations and C++ "auto".

		// FIXME [fix] >>
		//float data[240] = {0.0f,};  // 4 x 5 x 6 x 2 = 240.
		//Eigen::TensorMap<float, 3> result2(data, 4, 5, 6) = t4;
		//std::cout << result2(0, 0, 0) << std::endl;

		Eigen::TensorFixedSize<float, Eigen::Sizes<2, 3, 4>> result3 = t4;
		std::cout << result3(0, 0, 0) << std::endl;

		// One way to compute exp((t1 + t2) * 0.2f).
		auto t5 = t1 + t2;
		auto t6 = t5 * 0.2f;
		auto t7 = t6.exp();
		Eigen::Tensor<float, 3> result4 = t7;

		// Another way, exactly as efficient as the previous one.
		Eigen::Tensor<float, 3> result5 = ((t1 + t2) * 0.2f).exp();
	}

	{
		// Controlling when expression are evaluated.
		// There are several ways to control when expressions are evaluated:
		//	Assignment to a Tensor, TensorFixedSize, or TensorMap.
		//	Use of the eval() method.
		//	Assignment to a TensorRef.

		Eigen::Tensor<float, 3> t1(2, 3, 4);
		t1.setConstant(3);
		Eigen::Tensor<float, 3> t2(2, 3, 4);
		t2.setConstant(7);

		{
			// Assigning to a Tensor, TensorFixedSize, or TensorMap.

			auto t3 = t1 + t2;  // t3 is an Operation.
			auto t4 = t3 * 0.2f;  // t4 is an Operation.
			auto t5 = t4.exp();  // t5 is an Operation.
			Eigen::Tensor<float, 3> result1 = t5;  // The operations are evaluated.

			// We know that the result is a 4x4x2 tensor!
			Eigen::TensorFixedSize<float, Eigen::Sizes<4, 4, 2>> result2 = t5;
		}

		{
			// Calling eval().
			// When you compute large composite expressions, you sometimes want to tell Eigen that an intermediate value in the expression tree is worth evaluating ahead of time.
			// Semantically, calling eval() is equivalent to materializing the value of the expression in a temporary Tensor of the right size.
			// Note that the return value of eval() is itself an Operation.

			// The previous example could have been written.
			Eigen::Tensor<float, 3> result1 = ((t1 + t2) * 0.2f).exp();

			// If you want to compute (t1 + t2) once ahead of time you can write.
			Eigen::Tensor<float, 3> result2 = ((t1 + t2).eval() * 0.2f).exp();

			// .eval() knows the size!
			Eigen::TensorFixedSize<float, Eigen::Sizes<4, 4, 2>> tmp = t1 + t2;
			Eigen::Tensor<float, 3> result3 = (tmp * 0.2f).exp();

			// Here t3 is an evaluation Operation. t3 has not been evaluated yet.
			auto t3 = (t1 + t2).eval();

			// You can use t3 in another expression. Still no evaluation.
			auto t4 = (t3 * 0.2f).exp();

			// The value is evaluated when you assign the Operation to a Tensor, using an intermediate tensor to represent t3.x.
			Eigen::Tensor<float, 3> result4 = t4;
		}

		{
			// Assigning to a TensorRef.
			// If you need to access only a few elements from the value of an expression you can avoid materializing the value in a full tensor by using a TensorRef.
			// A TensorRef is a small wrapper class for any Eigen Operation. It provides overloads for the () operator that let you access individual values in the expression.

			// Create a TensorRef for the expression. The expression is not evaluated yet.
			Eigen::TensorRef<Eigen::Tensor<float, 3> > ref = ((t1 + t2) * 0.2f).exp();

			// Use "ref" to access individual elements. The expression is evaluated on the fly.
			float at_0 = ref(0, 0, 0);
			std::cout << ref(0, 1, 0) << std::endl;
		}

		{
			Eigen::Tensor<float, 2> A(2, 3), B(2, 3);
			A.setRandom();
			B.setRandom();

			Eigen::Tensor<float, 2> C = 2.0f * A + B.exp();

			std::cout << "A:\n" << A << std::endl;
			std::cout << "B:\n" << B << std::endl;
			std::cout << "C:\n" << C << std::endl;

			// Unary operation.
			Eigen::Tensor<float, 2> D = A.unaryExpr([](const float v) { return std::cos(v); });
			std::cout << "D:\n" << D << std::endl;

			// Binary operation.
			Eigen::Tensor<float, 2> E = A.binaryExpr(B, [](const float a, const float b) { return 2.0f * a + std::exp(b); });
			std::cout << "E:\n" << E << std::endl;
		}
	}

	//--------------------
	// Geometric operations.
	// Geometric operations result in tensors with different dimensions and, sometimes, sizes.
	// Examples of these operations are: reshape, pad, shuffle, stride, and broadcast.

	{
		Eigen::Tensor<float, 2> t(3, 4);
		t.setRandom();

		std::cout << "t:\n" << t << std::endl;
		std::cout << "t^T:\n" << my_transpose(t) << std::endl;
	}

	{
		// Reductions.

		Eigen::Tensor<float, 3> X(5, 2, 3);
		X.setRandom();

		std::cout << "X:\n" << X << std::endl;
		std::cout << "X.sum() = " << X.sum() << std::endl;
		std::cout << "X.mean() = " << X.mean() << std::endl;
		std::cout << "X.maximum() = " << X.maximum() << std::endl;
		std::cout << "X.minimum() = " << X.minimum() << std::endl;
		std::cout << "X.prod() = " << X.prod() << std::endl;
		std::cout << "X.any() = " << X.any() << std::endl;
		std::cout << "X.all() = " << X.all() << std::endl;

		Eigen::array<int, 2> dims({1, 2});
		std::cout << "X.sum(dims): " << X.sum(dims) << std::endl;
		std::cout << "X.maximum(dims): " << X.maximum(dims) << std::endl;
		std::cout << "X.reduce(dims) = " << X.reduce(dims, MyMaxReducer()) << std::endl;
	}

	{
		// Tensor convolutions.

		Eigen::Tensor<float, 4> input(1, 6, 6, 3);
		input.setRandom();
		Eigen::Tensor<float, 2> kernel(3, 3);
		kernel.setRandom();
		Eigen::Tensor<float, 4> output(1, 4, 4, 3);
		Eigen::array<int, 2> dims({1, 2});

		output = input.convolve(kernel, dims);

		std::cout << "Input:\n" << input << std::endl;
		std::cout << "Kernel:\n" << kernel << std::endl;
		std::cout << "Output:\n" << output << std::endl;
	}

	{
		// Softmax with tensors.

		Eigen::Tensor<float, 3> input(2, 4, 3);
		input.setValues({
			{{0.1, 1.0, -2.0}, {10.0, 2.0, 5.0}, {5.0, -5.0, 0.0}, {2.0, 3.0, 2.0}},
			{{100.0, 1000.0, -500.0}, {3.0, 3.0, 3.0}, {-1.0, 1.0, -1.0}, {-11.0, -0.2, -0.1}}
		});

		std::cout << "Input:\n" << input << std::endl;

		Eigen::Tensor<float, 3> output = my_softmax(input);
		std::cout << "Output:\n" << output << std::endl;
	}

	{
		// Reshape.

		Eigen::Tensor<float, 2> X(2, 3);
		X.setValues({{1, 2, 3}, {4, 5, 6}});

		std::cout << "X:\n"<< X << std::endl;
		std::cout << "Size of X = "<< X.size() << std::endl;

		Eigen::array<int, 3> new_dims({3,1,2});
		Eigen::Tensor<float, 3> Y = X.reshape(new_dims);

		std::cout << "Y:\n"<< Y << std::endl;
		std::cout << "Size of Y = "<< Y.size() << std::endl;
	}

	{
		// Broadcast.
		// Tensor.broadcast(bcast) repeats the tensor as many times as provided in the bcast parameter for each dimension.

		Eigen::Tensor<float, 2> Z(1, 3);
		Z.setValues({{1, 2, 3}});
		Eigen::array<int, 2> bcast({4, 2});
		Eigen::Tensor<float, 2> W = Z.broadcast(bcast);

		std::cout << "Z:\n"<< Z << std::endl;
		std::cout << "W:\n"<< W << std::endl;
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_eigen {

void tensor()
{
	local::simple_example();
}

}  // namespace my_eigen
