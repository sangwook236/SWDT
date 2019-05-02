#include <vector>
#include <functional>
#include <algorithm>
#include <iterator>
#include <iostream>


namespace {
namespace local {

void stl_algorithm_remove()
{
	const int arr[] = { 1, 3, 5, 3, 2, 0, 9, -2, -2, 9, 1 };
	std::vector<int> V(arr, arr + sizeof(arr) / sizeof(arr[0]));

	// erase-remove idiom
	V.erase(remove(V.begin(), V.end(), 3), V.end());
	std::copy(V.begin(), V.end(), std::ostream_iterator<int>(std::cout, " "));
	std::cout << std::endl;
}

void stl_algorithm_transform()
{
	// plus
	std::vector<int> V1(10, 5);
	std::transform(V1.begin(), V1.end(), V1.begin(), std::bind(std::plus<int>(), std::placeholders::_1, 7));
	std::copy(V1.begin(), V1.end(), std::ostream_iterator<int>(std::cout, " "));
	std::cout << std::endl; 

	// minus
	std::vector<int> V2(10, 2);
	std::transform(V2.begin(), V2.end(), V2.begin(), std::bind(std::minus<int>(), std::placeholders::_1, 7));
	std::copy(V2.begin(), V2.end(), std::ostream_iterator<int>(std::cout, " "));
	std::cout << std::endl; 

	// multiply
	std::vector<int> V3(10, 8);
	std::transform(V3.begin(), V3.end(), V3.begin(), std::bind(std::multiplies<int>(), std::placeholders::_1, 2));
	std::copy(V3.begin(), V3.end(), std::ostream_iterator<int>(std::cout, " "));
	std::cout << std::endl; 

	// divide
	std::vector<int> V4(10, 8);
	std::transform(V4.begin(), V4.end(), V4.begin(), std::bind(std::divides<int>(), std::placeholders::_1, 3));
	std::copy(V4.begin(), V4.end(), std::ostream_iterator<int>(std::cout, " "));
	std::cout << std::endl; 

	// square
	std::vector<int> V5(10, 8);
	std::transform(V5.begin(), V5.end(), V5.begin(), V5.begin(), std::multiplies<int>());
	std::copy(V5.begin(), V5.end(), std::ostream_iterator<int>(std::cout, " "));
	std::cout << std::endl; 
}

void stl_algorithm_unique()
{
	const int arr[] = { 1, 3, 5, 3, 2, 0, 9, -2, -2, 9, 1 };
	std::vector<int> V(arr, arr + sizeof(arr) / sizeof(arr[0]));

	std::sort(V.begin(), V.end());
	V.erase(std::unique(V.begin(), V.end()), V.end());
	std::copy(V.begin(), V.end(), std::ostream_iterator<int>(std::cout, " "));
	std::cout << std::endl;
}

void stl_algorithm_lower_bound_upper_bound()
{
	const int myints[] = { 10,20,30,30,20,10,10,20 };
	std::vector<int> v(myints, myints + 8);  // 10 20 30 30 20 10 10 20

	std::sort(v.begin(), v.end());  // 10 10 10 20 20 20 30 30

	{
		std::vector<int>::iterator low = std::lower_bound(v.begin(), v.end(), 20);
		std::vector<int>::iterator up = std::upper_bound(v.begin(), v.end(), 20);

		std::cout << "lower_bound at position: " << (size_t)std::distance(v.begin(), low) << std::endl;
		std::cout << "upper_bound at position: " << (size_t)std::distance(v.begin(), up) << std::endl;
	}

	//
	{
		std::vector<int>::iterator low = std::lower_bound(v.begin(), v.end(), 0);
		std::vector<int>::iterator up = std::upper_bound(v.begin(), v.end(), 40);

		std::cout << "lower_bound at position: " << (size_t)std::distance(v.begin(), low) << std::endl;
		std::cout << "upper_bound at position: " << (size_t)std::distance(v.begin(), up) << std::endl;
	}
}

}  // namespace local
}  // unnamed namespace

void stl_algorithm()
{
	std::cout << ">>> stl algorithm: remove" << std::endl;
	local::stl_algorithm_remove();
	std::cout << "\n>>> stl algorithm: transform" << std::endl;
	local::stl_algorithm_transform();
	std::cout << "\n>>> stl algorithm: unique" << std::endl;
	local::stl_algorithm_unique();
	std::cout << "\n>>> stl algorithm: lower_bound & upper_bound" << std::endl;
	local::stl_algorithm_lower_bound_upper_bound();
}
