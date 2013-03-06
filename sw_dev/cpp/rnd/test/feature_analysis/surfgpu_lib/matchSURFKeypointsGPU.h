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

#ifndef MATCHSURFKEYPOINTSGPU_H
#define MATCHSURFKEYPOINTSGPU_H

/**	\brief Set threshold for matching
 *	\param threshold threshold for matching
 *
 *	This function must be called once before calling matchSURFKeypointsGPU().
 */
void prepare_matchSURFKeypointsGPU(float threshold);

/**	\brief Match SURF keypoints from 2 given sets
 *	\param d_result device pointer to save matched indices to
 *	\param d_dist device pointer to save smallest distance to
 *	\param d_set1 device pointer to first interest point set
 *	\param num_points_set1 number of interest points in first set
 *	\param set1_pitch number of bytes between two consecutive interest points
 *	\param d_set2 device pointer to second interest point set
 *	\param num_points_set2 number of interest points in second set
 *	\param set2_pitch number of bytes between two consecutive interest points
 *	\param desc_len descriptor length in elements
 *
 *	prepare_matchSURFKeypointsGPU() must be called once before calling this
 *	function.
 */
void matchSURFKeypointsGPU(
	int *d_result, float *d_dist,
	float *d_set1, size_t num_points_set1, size_t set1_pitch,
	float *d_set2, size_t num_points_set2, size_t set2_pitch,
	unsigned int desc_len);

#endif /* MATCHSURFKEYPOINTSGPU_H */
