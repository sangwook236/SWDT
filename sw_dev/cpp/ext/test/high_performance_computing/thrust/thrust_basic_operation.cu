#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <iostream>
#include <vector>
#include <list>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_thrust {

void basic_operation()
{
	//-----------------------------------------------------
	{
		const std::size_t N = 10;

		// Raw pointer to device memory.
		int *raw_dev_ptr = NULL;
		cudaMalloc((void **)&raw_dev_ptr, N * sizeof(int));

		// Wrap raw pointer with a device_ptr.
		thrust::device_ptr<int> dev_ptr(raw_dev_ptr);

		// Use device_ptr in thrust algorithms.
		thrust::fill(dev_ptr, dev_ptr + N, 0);
	}

	//-----------------------------------------------------
	{
		const std::size_t N = 10;

		// Create a device_ptr.
		thrust::device_ptr<int> dev_ptr = thrust::device_malloc<int>(N);

		// Use device_ptr in thrust algorithms.
		thrust::fill(dev_ptr, dev_ptr + N, 1);

		// EXtract raw pointer from device_ptr.
		int *raw_dev_ptr = thrust::raw_pointer_cast(dev_ptr);
	}
}

void vector()
{
	//-----------------------------------------------------
	{
		// H has storage for 4 integers.
		thrust::host_vector<int> H(4);

		// Initialize individual elements.
		H[0] = 14;
		H[1] = 20;
		H[2] = 38;
		H[3] = 46;

		// H.size() returns the size of vector H.
		std::cout << "H has size " << H.size() << std::endl;

		// Print contents of H.
		for (int i = 0; i < H.size(); ++i)
			std::cout << "H[" << i << "] = " << H[i] << std::endl;

		// Resize H.
		H.resize(2);
		std::cout << "H now has size " << H.size() << std::endl;
	
		// Copy host_vector H to device_vector D.
		thrust::device_vector<int> D = H;

		// Elements of D can be modified.
		D[0] = 99;
		D[1] = 88;

		// Print contents of D.
		//for (int i = 0; i < D.size(); ++i)
		//	std::cout << "D[" << i << "] = " << D[i] << std::endl;
		int i = 0;
		for (thrust::device_vector<int>::iterator it = D.begin(); it < D.end(); ++it, ++i)
			std::cout << "D[" << i << "] = " << *it << std::endl;

		// H and D are automatically deleted when the function returns.
	}

	//-----------------------------------------------------
	{
		// Initialize all ten integers of a device_vector to 1.
		thrust::device_vector<int> D(10, 1);

		// Set the first seven elements of a vector to 9.
		thrust::fill(D.begin(), D.begin() + 7, 9);

		// Initialize a host_vector with the first five elements of D.
		thrust::host_vector<int> H(D.begin(), D.begin() + 5);

		// Set the elements of H to 0, 1, 2, 3, ...
		thrust::sequence(H.begin(), H.end());
	
		// Copy all of H back to the beginning of D.
		thrust::copy(H.begin(), H.end(), D.begin());
	
		// Print D.
		for (int i = 0; i < D.size(); ++i)
			std::cout << "D[" << i << "] = " << D[i] << std::endl;
	}
}

void list()
{
	//-----------------------------------------------------
	{
		// Create an STL list.
		std::list<int> stl_list;
		stl_list.push_back(10);
		stl_list.push_back(20);
		stl_list.push_back(30);
		stl_list.push_back(40);

		// Initialize a device_vector with the list.
		thrust::device_vector<int> D(stl_list.begin(), stl_list.end());

		// Copy a device_vector into an STL vector.
		std::vector<int> stl_vector(D.size());
		thrust::copy(D.begin(), D.end(), stl_vector.begin());
	}
}

}  // namespace my_thrust
