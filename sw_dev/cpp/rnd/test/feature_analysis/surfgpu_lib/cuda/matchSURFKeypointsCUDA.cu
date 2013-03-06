/*
 * Copyright (C) 2009-2010 Andre Schulz, Florian Jung, Sebastian Hartte,
 *						   Daniel Trick, Christan Wojek, Konrad Schindler,
 *						   Jens Ackermann, Michael Goesele
 * Copyright (C) 2008-2009 Christopher Evans <chris.evans@irisys.co.uk>, MSc University of Bristol
 *
 * This file is part of SURFGPU.
 *
 * SURFGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SURFGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SURFGPU.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef CUDA_MATCHSURFKEYPOINTSCUDA_CU
#define CUDA_MATCHSURFKEYPOINTSCUDA_CU

#ifdef __DEVICE_EMULATION__
#	include <stdio.h>
#endif
#include <float.h>
#include "common_kernel.h"
#include "reductionCUDA.cu"


__constant__ float c_threshold;

/**	\brief Match SURF keypoints between 2 given sets
 *	\param g_result pointer for saving indices from set 2
 *	\param g_dist pointer for saving smallest distance for points from set 1
 *	\param g_set1 pointer to 1st set of SURF keypoints
 *	\param num_points_set1 number of keypoints in 1st set
 *	\param set1_pitch number of elements between two consecutive keypoints
 *	\param g_set2 pointer to 2nd set of SURF keypoints
 *	\param num_points_set2 number of points in 2nd set
 *	\param set2_pitch number of elements between two consecutive keypoints
 *
 *	Hardcoded for a descriptor length of 64 elements.
 *	Metric for a match is smallest euclidean distance of the descriptors.
 *
 *	Execution configuration:
 *	  Thread block: { 32, 4, 1 }
 *	  Block grid  : { ceil(num_points_set1, block.x), 1 }
 *
 *	To cut down on global memory bandwidth usage the kernel processes 4 points
 *	from the 1st set in parallel. The thread block size in X is expected to be
 *	equal to the warp size to reduce the number of __syncthreads() calls. For a
 *	warp size of 32 this means that each thread keeps 2 elements of a point's
 *	descriptor in registers. Then, the first 64 threads load the descriptor
 *	elements of a point from the 2nd set into shared memory so that the other
 *	threads can compute the euclidean distance to that point.
 */
__global__ void
matchSURFKeypointsCUDA(
	int *g_result, float *g_dist,
	float *g_set1, size_t num_points_set1, size_t set1_pitch,
	float *g_set2, size_t num_points_set2, size_t set2_pitch)
{
	__shared__ float s_diff_vec[256];
	__shared__ float s_temp[64];
	float *s_blk_diff_vec = s_diff_vec + IMUL(threadIdx.y, 64);
	int best_idx = -1;
	float best_dist = FLT_MAX;

	// Load descriptor element from 1st set for each thread
	unsigned int bid = IMUL(blockIdx.x, blockDim.y) + threadIdx.y;
	size_t load_idx = IMUL(bid, set1_pitch) + threadIdx.x;

	float set1_desc_elt1;
	float set1_desc_elt2;
	if (bid < num_points_set1)
	{
		set1_desc_elt1 = g_set1[load_idx];
		load_idx += blockDim.x;
		set1_desc_elt2 = g_set1[load_idx];
	}

	unsigned int tid = IMUL(threadIdx.y, blockDim.x) + threadIdx.x;
	for (size_t i = 0; i < num_points_set2; i++)
	{
		/* Load descriptor element from 2nd set for each thread
		 * and write it to shared memory.
		 */
		load_idx = IMUL(i, set2_pitch) + tid;
		float set2_desc_elt;
		if (tid < 64)
		{
			set2_desc_elt = g_set2[load_idx];
			s_temp[tid] = set2_desc_elt;
		}
		__syncthreads();

		unsigned int smem_idx = threadIdx.x;
		set2_desc_elt = s_temp[smem_idx];

		/* Compute difference between elements */
		float elt_diff = set1_desc_elt1 - set2_desc_elt;

		/* Square the difference. Needed for computing the euclidean distance
		 * of the difference vector.
		 */
		s_blk_diff_vec[threadIdx.x] = elt_diff * elt_diff;

		/* Same for the 2nd element. */
		smem_idx += blockDim.x;
		set2_desc_elt = s_temp[smem_idx];
		elt_diff = set1_desc_elt2 - set2_desc_elt;
		s_blk_diff_vec[smem_idx] = elt_diff * elt_diff;

		// Sum up squared elements
		reduction64OptCUDA(s_blk_diff_vec, threadIdx.x);
		__syncthreads();

		float cur_dist = sqrtf(s_blk_diff_vec[0]);
		if (cur_dist < best_dist)
		{
			best_dist = cur_dist;
			best_idx = i;
		}
	}

	if (threadIdx.x > 0
		|| bid >= num_points_set1) return;

	g_dist[bid] = best_dist;
	if (best_dist >= c_threshold)
	{
		best_idx = -1;
	}
	g_result[bid] = best_idx;
}

/**	\brief Match SURF keypoints between 2 given sets
 *	\param g_result pointer for saving indices from set 2
 *	\param g_dist pointer for saving smallest distance for points from set 1
 *	\param g_set1 pointer to 1st set of SURF keypoints
 *	\param num_points_set1 number of keypoints in 1st set
 *	\param set1_pitch number of elements between two consecutive keypoints
 *	\param g_set2 pointer to 2nd set of SURF keypoints
 *	\param num_points_set2 number of points in 2nd set
 *	\param set2_pitch number of elements between two consecutive keypoints
 *	\param desc_len descriptor length (in number of elements)
 *
 *	Hardcoded for a descriptor length of 64 elements.
 *	Metric for a match is the ratio of distance of the smallest to the second
 *	smallest euclidean distance of the descriptors.
 *
 *	Execution configuration:
 *	  Thread block: { 32, 4, 1 }
 *	  Block grid  : { ceil(num_points_set1, block.x), 1 }
 *
 *	Pretty much the same as matchSURFKeypointsCUDA() except for the matching
 *	metric.
 */
__global__ void
matchSURFKeypoints2CUDA(
	int *g_result, float *g_dist,
	float *g_set1, size_t num_points_set1, size_t set1_pitch,
	float *g_set2, size_t num_points_set2, size_t set2_pitch)
{
	__shared__ float s_diff_vec[256];
	__shared__ float s_temp[64];
	float *s_blk_diff_vec = s_diff_vec + IMUL(threadIdx.y, 64);
	int best_idx = -1;
	float best_dist = FLT_MAX;
	float better_dist = FLT_MAX;

	// Load descriptor element from 1st set for each thread
	unsigned int bid = IMUL(blockIdx.x, blockDim.y) + threadIdx.y;
	size_t load_idx = IMUL(bid, set1_pitch) + threadIdx.x;

	float set1_desc_elt1;
	float set1_desc_elt2;
	if (bid < num_points_set1)
	{
		set1_desc_elt1 = g_set1[load_idx];
		load_idx += blockDim.x;
		set1_desc_elt2 = g_set1[load_idx];
	}

	unsigned int tid = IMUL(threadIdx.y, blockDim.x) + threadIdx.x;
	for (size_t i = 0; i < num_points_set2; i++)
	{
		// Load descriptor element from 2nd set for each thread
		// and write to shared memory.
		load_idx = IMUL(i, set2_pitch) + tid;
		float set2_desc_elt;
		if (tid < 64)
		{
			set2_desc_elt = g_set2[load_idx];
			s_temp[tid] = set2_desc_elt;
		}
		__syncthreads();

		unsigned int smem_idx = threadIdx.x;
		set2_desc_elt = s_temp[smem_idx];

		// Compute difference between elements
		float elt_diff = set1_desc_elt1 - set2_desc_elt;

		// Compute euclidean distance
		// Square each element of the difference vector and write result to
		// shared mem.
		s_blk_diff_vec[threadIdx.x] = elt_diff * elt_diff;

		// And the same for the 2nd element
		smem_idx += blockDim.x;
		set2_desc_elt = s_temp[smem_idx];

		elt_diff = set1_desc_elt2 - set2_desc_elt;
		s_blk_diff_vec[blockDim.x + threadIdx.x] = elt_diff * elt_diff;

		// Sum up squared elements
		reduction64OptCUDA(s_blk_diff_vec, threadIdx.x);
		__syncthreads();

		float cur_dist = sqrtf(s_blk_diff_vec[0]);
		if (cur_dist < best_dist)
		{
			better_dist = best_dist;

			best_dist = cur_dist;
			best_idx = i;
		}
		else if (cur_dist < better_dist)
		{
			better_dist = cur_dist;
		}
	}

	if (threadIdx.x > 0
		|| bid >= num_points_set1) return;

	float ratio = best_dist / better_dist;
	g_dist[bid] = ratio;
	if (ratio >= c_threshold)
	{
		best_idx = -1;
	}
	g_result[bid] = best_idx;
}


#endif /* CUDA_MATCHSURFKEYPOINTSCUDA_CU */
