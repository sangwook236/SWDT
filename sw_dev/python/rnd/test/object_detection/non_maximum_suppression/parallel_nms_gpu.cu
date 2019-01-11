#include <iostream>
#include <cuda_runtime.h>
#include "parallel_nms.h"

#define CUDA_CHECK(err) assert_cuda((err), __FILE__, __LINE__)

inline void assert_cuda(cudaError_t err, const char *file, int line, bool abort=true)
{
	if (err != cudaSuccess)
	{
		std::cout << "assert_cuda: " << cudaGetErrorString(err) << file << line << std::endl;
		if (abort) exit(err);
	}
}

__device__ float IOUcalc(const box &b1, const box &b2)
{
	const float ai = float(b1.w + 1) * (b1.h + 1);
	const float aj = float(b2.w + 1) * (b2.h + 1);

	const float x_inter = max(b1.x, b2.x);
	const float y_inter = max(b1.y, b2.y);
	const float x2_inter = min((b1.x + b1.w), (b2.x + b2.w));
	const float y2_inter = min((b1.y + b1.h), (b2.y + b2.h));

	const float w = max(0.0f, x2_inter - x_inter);
	const float h = max(0.0f, y2_inter - y_inter);

	return ((w * h) / (ai + aj - w * h));
}

__global__ void NMS_GPU(box *d_b, float threshold, bool *d_res)
{
	int abs_y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int abs_x = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (d_b[abs_x].s >= d_b[abs_y].s && IOUcalc(d_b[abs_x], d_b[abs_y]) >= threshold)
		d_res[abs_y] = false;
}

void parallel_nms_gpu(box *b, int count, float threshold, bool *res)
{
	box *d_b;
	bool *d_res;

	CUDA_CHECK(cudaMalloc((void**)&d_res, count*sizeof(bool)));
	CUDA_CHECK(cudaMemcpy(d_res, res, sizeof(bool) * count, cudaMemcpyHostToDevice));

	CUDA_CHECK(cudaMalloc((void**)&d_b, sizeof(box) * count));
	CUDA_CHECK(cudaMemcpy(d_b, b, sizeof(box) * count, cudaMemcpyHostToDevice));

	NMS_GPU<<<dim3(1, count, 1), count>>>(d_b, threshold, d_res);

	cudaThreadSynchronize();

	CUDA_CHECK(cudaMemcpy(res, d_res, sizeof(bool) * count, cudaMemcpyDeviceToHost));
}
