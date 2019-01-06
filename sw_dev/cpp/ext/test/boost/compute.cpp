//#include <boost/compute/core.hpp>
#include <boost/compute.hpp>
#include <boost/smart_ptr.hpp>
#include <CL/cl.hpp>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <vector>
#include <complex>
#include <chrono>


namespace {
namespace local {

// REF [site] >> http://www.boost.org/doc/libs/1_61_0/libs/compute/doc/html/boost_compute/tutorial.html
void device_info()
{
	// Get the default device.
	boost::compute::device device = boost::compute::system::default_device();

	std::cout << "Device: " << device.name() << std::endl;

	// Devices.
	const size_t numDevices = boost::compute::system::device_count();
	std::cout << "#devices = " << numDevices << std::endl;

	const std::vector<boost::compute::device> &devices = boost::compute::system::devices();
	for (const auto dev: devices)
		std::cout << "\tDevice: " << dev.name() << std::endl;

	boost::compute::device device0 = boost::compute::system::find_device(devices[0].name());

	// Platform.
	const size_t numPlatforms = boost::compute::system::platform_count();
	std::cout << "#platforms = " << numPlatforms << std::endl;

	const std::vector<boost::compute::platform> &platforms = boost::compute::system::platforms();
	for (const auto plat: platforms)
		std::cout << "\tPlatform: " << plat.name() << std::endl;
}

// REF [site] >> http://www.boost.org/doc/libs/1_61_0/libs/compute/doc/html/boost_compute/tutorial.html
void transfer_data(boost::compute::context &context, boost::compute::command_queue &queue)
{
	// Create data array on host.
	int host_data[] = { 1, 3, 5, 7, 9 };

	// Create vector on device.
	boost::compute::vector<int> device_vector(5, context);

	// Copy from host to device.
	boost::compute::copy(host_data, host_data + 5, device_vector.begin(), queue);

	// Create vector on host.
	std::vector<int> host_vector(5);

	// Copy data back to host.
	boost::compute::copy(device_vector.begin(), device_vector.end(), host_vector.begin(), queue);
}

// REF [site] >> http://www.boost.org/doc/libs/1_61_0/libs/compute/doc/html/boost_compute/tutorial.html
void transform_data(boost::compute::context &context, boost::compute::command_queue &queue)
{
	// Generate random data on the host.
	std::vector<float> host_vector(10000);
	std::generate(host_vector.begin(), host_vector.end(), rand);

	// Create a vector on the device.
	boost::compute::vector<float> device_vector(host_vector.size(), context);

	// Transfer data from the host to the device.
	boost::compute::copy(host_vector.begin(), host_vector.end(), device_vector.begin(), queue);

	// Calculate the square-root of each element in-place.
	boost::compute::transform(device_vector.begin(), device_vector.end(), device_vector.begin(), boost::compute::sqrt<float>(), queue);

	// Copy values back to the host.
	boost::compute::copy(device_vector.begin(), device_vector.end(), host_vector.begin(), queue);
}

// REF [site] >> http://www.boost.org/doc/libs/1_61_0/libs/compute/doc/html/boost_compute/advanced_topics.html
void vector_data_type(boost::compute::context &context, boost::compute::command_queue &queue)
{
	// Point coordinates.
	float points[] = {
		1.0f, 2.0f, 3.0f, 0.0f,
		-2.0f, -3.0f, 4.0f, 0.0f,
		1.0f, -2.0f, 2.5f, 0.0f,
		-7.0f, -3.0f, -2.0f, 0.0f,
		3.0f, 4.0f, -5.0f, 0.0f
	};

	// Create vector for five points.
	boost::compute::vector<boost::compute::float4_> vector(5, context);

	// Copy point data to the device.
	boost::compute::copy(
		reinterpret_cast<boost::compute::float4_ *>(points),
		reinterpret_cast<boost::compute::float4_ *>(points) + 5,
		vector.begin(),
		queue
	);

	// Calculate sum.
	const boost::compute::float4_ sum = boost::compute::accumulate(vector.begin(), vector.end(), boost::compute::float4_(0, 0, 0, 0), queue);

	// Calculate centroid.
	boost::compute::float4_ centroid;
	for (size_t i = 0; i < 4; ++i)
		centroid[i] = sum[i] / 5.0f;

	// Print centroid.
	std::cout << "Centroid: " << centroid << std::endl;
}

// REF [site] >> http://www.boost.org/doc/libs/1_61_0/libs/compute/doc/html/boost_compute/advanced_topics.html
void complex_type()
{
	// Create vector on device.
	boost::compute::vector<std::complex<float>> vector;

	// Insert two complex values.
	vector.push_back(std::complex<float>(1.0f, 3.0f));
	vector.push_back(std::complex<float>(2.0f, 4.0f));
}

#if 0
boost::compute::function<int (int)> add_four = boost::compute::make_function_from_source<int (int)>(
	"add_four",
	"int add_four(int x) { return x + 4; }"
);
#else
BOOST_COMPUTE_FUNCTION(int, add_four, (int x),
{
	return x + 4;
});
#endif

// REF [site] >> http://www.boost.org/doc/libs/1_61_0/libs/compute/doc/html/boost_compute/advanced_topics.html
// REF [site] >> http://kylelutz.blogspot.kr/2014/03/custom-opencl-functions-in-c-with.html
void custom_function(boost::compute::context &context, boost::compute::command_queue &queue)
{
	// Create data array on host.
	std::vector<float> host_vector = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

	// Create a vector on the device.
	boost::compute::vector<float> device_vector(host_vector.size(), context);

	// Transfer data from the host to the device.
	boost::compute::copy(host_vector.begin(), host_vector.end(), device_vector.begin(), queue);

	boost::compute::transform(device_vector.begin(), device_vector.end(), device_vector.begin(), add_four, queue);

	// Copy values back to the host.
	boost::compute::copy(device_vector.begin(), device_vector.end(), host_vector.begin(), queue);
}

// REF [site] >> http://www.boost.org/doc/libs/1_61_0/libs/compute/doc/html/boost_compute/advanced_topics.html
void lambda_expression(boost::compute::context &context, boost::compute::command_queue &queue)
{
	// Create data array on host.
	std::vector<float> host_vector = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

	// Create a vector on the device.
	boost::compute::vector<float> device_vector(host_vector.size(), context);
	boost::compute::copy(host_vector.begin(), host_vector.end(), device_vector.begin(), queue);

	boost::compute::count_if(device_vector.begin(), device_vector.end(), boost::compute::lambda::_1 % 2 == 1, queue);
	boost::compute::transform(device_vector.begin(), device_vector.end(), device_vector.begin(), boost::compute::lambda::_1 * 3 - 4, queue);

	boost::compute::function<int (int)> add_four = boost::compute::lambda::_1 + 4;
}

// REF [site] >> http://www.boost.org/doc/libs/1_61_0/libs/compute/doc/html/boost_compute/advanced_topics.html
void asynchronous_operation(boost::compute::context &context, boost::compute::command_queue &queue)
{
	// Create data array on host.
	//std::array<float, 10> host_vector = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };  // NOTICE [info] >> Run-time error.
	std::vector<float> host_vector = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

	// Create a vector on the device.
	boost::compute::vector<float> device_vector(host_vector.size(), context);
	boost::compute::copy(host_vector.begin(), host_vector.end(), device_vector.begin(), queue);

	// Copy data to the device asynchronously.
	boost::compute::future<void> f = boost::compute::copy_async(host_vector.begin(), host_vector.end(), device_vector.begin(), queue);

	// Perform other work on the host or device.
	// ...

	// Ensure the copy is completed.
	f.wait();

	// Use data on the device (e.g. sort).
	boost::compute::sort(device_vector.begin(), device_vector.end(), queue);
}

// REF [site] >> http://www.boost.org/doc/libs/1_61_0/libs/compute/doc/html/boost_compute/advanced_topics.html
void performance_measure(boost::compute::context &context, boost::compute::command_queue &queue)
{
	// Generate random data on the host.
	std::vector<int> host_vector(16000000);
	std::generate(host_vector.begin(), host_vector.end(), rand);

	// Create a vector on the device.
	boost::compute::vector<int> device_vector(host_vector.size(), context);

	// Copy data from the host to the device.
	boost::compute::future<void> future = boost::compute::copy_async(host_vector.begin(), host_vector.end(), device_vector.begin(), queue);

	// Wait for copy to finish.
	future.wait();

	// Get elapsed time from event profiling information.
	boost::chrono::milliseconds duration = future.get_event().duration<boost::chrono::milliseconds>();

	// Print elapsed time in milliseconds.
	std::cout << "Time: " << duration.count() << " ms." << std::endl;
}

// REF [site] >> http://www.boost.org/doc/libs/1_61_0/libs/compute/doc/html/boost_compute/advanced_topics.html
void opencl_api_interoperability(boost::compute::context &ctx)
{
	// Query number of devices using the OpenCL API.
	cl_uint num_devices;
	clGetContextInfo(ctx, CL_CONTEXT_NUM_DEVICES, sizeof(cl_uint), &num_devices, 0);
	std::cout << "#devices: " << num_devices << std::endl;
}

void opencl_cpp_wrapper_example()
{
	static char kernelSourceCode[] = "\
		__kernel void vadd(__global int *a, __global int *b, __global int *c)\
		{\
			size_t i = get_global_id(0);\
			\
			c[i] = a[i] + b[i];\
		}\
		";

	size_t const BUFFER_SIZE = 1UL << 13;
	boost::scoped_array<int> A(new int[BUFFER_SIZE]);
	boost::scoped_array<int> B(new int[BUFFER_SIZE]);
	boost::scoped_array<int> C(new int[BUFFER_SIZE]);

	std::iota(A.get(), A.get() + BUFFER_SIZE, 0);
	std::transform(A.get(), A.get() + BUFFER_SIZE, B.get(), std::bind(std::multiplies<int>(), std::placeholders::_1, 2));

	try
	{
		const auto start = std::chrono::high_resolution_clock::now();

		std::vector<cl::Platform> platformList;
		// Pick platform.
		cl::Platform::get(&platformList);
		// Pick first platform.
		cl_context_properties cprops[] = {
			CL_CONTEXT_PLATFORM, (cl_context_properties)(platformList[0])(), 0
		};
		cl::Context context(CL_DEVICE_TYPE_GPU, cprops);
		// Query the set of devices attached to the context.
		std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
		// Create command-queue.
		cl::CommandQueue queue(context, devices[0], 0);

		// Create the program from source.
		cl::Program::Sources sources(1, std::make_pair(kernelSourceCode, 0));
		cl::Program program(context, sources);
		// Build program.
		program.build(devices);

		// Create buffer for A and copy host contents.
		cl::Buffer aBuffer = cl::Buffer(
			context,
			CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			BUFFER_SIZE * sizeof(int),
			(void *)&A[0]
		);
		// Create buffer for B and copy host contents.
		cl::Buffer bBuffer = cl::Buffer(
			context,
			CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			BUFFER_SIZE * sizeof(int),
			(void *)&B[0]
		);
		// Create buffer that uses the host ptr C.
		cl::Buffer cBuffer = cl::Buffer(
			context,
			CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
			BUFFER_SIZE * sizeof(int),
			(void *)&C[0]
		);

		// Create kernel object.
		cl::Kernel kernel(program, "vadd");
		// Set kernel args.
		kernel.setArg(0, aBuffer);
		kernel.setArg(1, bBuffer);
		kernel.setArg(2, cBuffer);

		// Do the work.
		void *output;
		{
			queue.enqueueNDRangeKernel(
				kernel,
				cl::NullRange,
				cl::NDRange(BUFFER_SIZE),
				cl::NullRange
			);
			output = (int *)queue.enqueueMapBuffer(
				cBuffer,
				CL_TRUE, // block
				CL_MAP_READ,
				0,
				BUFFER_SIZE * sizeof(int)
			);
		}
		//std::ofstream gpu("opencl_cpp_wrapper.txt");
		//for (int i = 0; i < BUFFER_SIZE; ++i)
		//	gpu << C[i] << " ";
		queue.enqueueUnmapMemObject(cBuffer, output);

		std::cout << "OpenCL C++ Wrapper took " << std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count() << " ms." << std::endl;
	}
	catch (const cl::Error const &ex)
	{
		std::cerr << ex.what() << std::endl;
	}
}

void boost_compute_example()
{
	size_t const BUFFER_SIZE = 1UL << 13;
	boost::scoped_array<int> A(new int[BUFFER_SIZE]), B(new int[BUFFER_SIZE]), C(new int[BUFFER_SIZE]);

	std::iota(A.get(), A.get() + BUFFER_SIZE, 0);
	std::transform(A.get(), A.get() + BUFFER_SIZE, B.get(), std::bind(std::multiplies<int>(), std::placeholders::_1, 2));

	try
	{
		if (boost::compute::system::default_device().type() != CL_DEVICE_TYPE_GPU)
		{
			std::cerr << "No GPU." << std::endl;
		}
		else
		{
			const auto start = std::chrono::high_resolution_clock::now();

			boost::compute::command_queue queue = boost::compute::system::default_queue();
			boost::compute::mapped_view<int> mA(static_cast<const int *>(A.get()), BUFFER_SIZE), mB(static_cast<const int *>(B.get()), BUFFER_SIZE);
			boost::compute::mapped_view<int> mC(C.get(), BUFFER_SIZE);
			{
				boost::compute::transform(
					mA.cbegin(), mA.cend(),
					mB.cbegin(),
					mC.begin(),
					boost::compute::plus<int>(),
					queue
				);
				mC.map(CL_MAP_READ, queue);
			}
			//std::ofstream gpu("boost_compute.txt");
			//for (size_t i = 0; i != BUFFER_SIZE; ++i)
			//	gpu << C[i] << " ";
			mC.unmap(queue);

			std::cout << "Boost.Compute took " << std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count() << " ms." << std::endl;
		}
	}
	catch (const boost::compute::opencl_error const &ex)
	{
		std::cerr << ex.what() << std::endl;
	}
}
}  // namespace local
}  // unnamed namespace

void compute()
{
	// REF [site] >> http://www.boost.org/doc/libs/1_61_0/libs/compute/doc/html/boost_compute/reference.html

	local::device_info();

	{
		boost::compute::device device = boost::compute::system::default_device();
		boost::compute::context context(device);
		boost::compute::command_queue queue(context, device);

		local::transfer_data(context, queue);
		local::transform_data(context, queue);

		local::vector_data_type(context, queue);
		local::complex_type();

		local::custom_function(context, queue);
		//local::lambda_expression(context, queue);  // NOTICE [error] >> Run-time error.

		local::asynchronous_operation(context, queue);
	}

	{
		boost::compute::device device = boost::compute::system::default_device();
		boost::compute::context context(device);
		boost::compute::command_queue queue(context, device, boost::compute::command_queue::enable_profiling);

		local::performance_measure(context, queue);
	}

	{
		// Create a context object.
		boost::compute::context context = boost::compute::system::default_context();

		// REF [site] >> http://www.boost.org/doc/libs/1_61_0/libs/compute/doc/html/boost_compute/porting_guide.html
		local::opencl_api_interoperability(context);
		// Boost.Compute provides a number of facilities to ease interoperability with other C and C++ libraries including OpenGL, OpenCV, Eigen, Qt, and VTK.
		//	REF [site] >> http://www.boost.org/doc/libs/1_61_0/libs/compute/doc/html/boost_compute/interop.html
	}

	// Performance: Boost.Compute vs. OpenCL C++ Wrapper.
	//	REF [site] >> https://stackoverflow.com/questions/23901979/performance-boost-compute-v-s-opencl-c-wrapper
	{
		local::opencl_cpp_wrapper_example();
		local::boost_compute_example();
	}
}
