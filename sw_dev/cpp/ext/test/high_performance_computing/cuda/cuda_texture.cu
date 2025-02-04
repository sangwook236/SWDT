#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
//#include <texture_indirect_functions.h>


namespace {
namespace local {

#if defined(__CUDACC__) && __CUDACC_VER_MAJOR__ < 12
// When using texture references

const size_t N = 1024;
texture<float, 1, cudaReadModeElementType> tex;

// Texture reference name must be known at compile time
__global__ void kernel()
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	float x = tex1Dfetch(tex, i);
	// Do some work using x...
}

void call_kernel(float *buffer)
{
	// Bind texture to buffer
	cudaBindTexture(0, tex, buffer, N * sizeof(float));

	dim3 block(128, 1, 1);
	dim3 grid(N / block.x, 1, 1);
	kernel<<<grid, block>>>();

	// Unbind texture from buffer
	cudaUnbindTexture(tex);
}

void simple_texture_test()
{
	// Declare and allocate memory
	float *buffer = nullptr;
	cudaMalloc(&buffer, N * sizeof(float));

	call_kernel(buffer);

	cudaFree(buffer);
	buffer = nullptr;
}
#else
// When using texture objects

const size_t N = 1024;

// Texture object is a kernel argument
__global__ void kernel(cudaTextureObject_t tex)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	float x = tex1Dfetch<float>(tex, i);
	// Do some work using x ...
}

void call_kernel(cudaTextureObject_t tex)
{
	dim3 block(128, 1, 1);
	dim3 grid(N / block.x, 1, 1);
	kernel<<<grid, block>>>(tex);
}

void simple_texture_test()
{
	// Declare and allocate memory
	float *buffer = nullptr;
	cudaMalloc(&buffer, N * sizeof(float));

	// Create a resource description
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeLinear;
	resDesc.res.linear.devPtr = buffer;
	//resDesc.res.linear.desc = cudaCreateChannelDesc<float>();
	resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
	resDesc.res.linear.desc.x = 32;  // Bits per channel
	resDesc.res.linear.sizeInBytes = N * sizeof(float);

	// Create a texture description
	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.readMode = cudaReadModeElementType;

	// Create a texture object: we only have to do this once!
	cudaTextureObject_t tex = 0;
	cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);

	call_kernel(tex);  // Pass texture as argument

	// Destroy the texture object
	cudaDestroyTextureObject(tex);

	cudaFree(buffer);
	buffer = nullptr;
}
#endif

}  // namespace local
}  // unnamed namespace

namespace my_cuda {

void texture_test()
{
	// REF [site] >> https://developer.nvidia.com/blog/cuda-pro-tip-kepler-texture-objects-improve-performance-and-flexibility/
	local::simple_texture_test();

	std::cout << "Texture tested." << std::endl;
}

}  // namespace my_cuda
