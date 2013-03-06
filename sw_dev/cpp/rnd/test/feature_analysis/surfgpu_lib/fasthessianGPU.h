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

#ifndef FASTHESSIAN_H
#define FASTHESSIAN_H

#include <vector>

#include "cv.h"
//--S [] 2013/03/06: Sang-Wook Lee
//#include "ipoint.h"
#include "ipointGPU.h"
//--E [] 2013/03/06: Sang-Wook Lee
#include "fasthessian_cudaipoint.h"

static const int OCTAVES = 4;
static const int INTERVALS = 4;
static const float THRES = 0.0004f;
static const int INIT_SAMPLE = 2;

//--S [] 2013/03/06: Sang-Wook Lee
namespace surfgpu {
//--E [] 2013/03/06: Sang-Wook Lee


class FastHessian {

  public:

    //! Destructor
    ~FastHessian();

    //! Constructor without image
    FastHessian(std::vector<Ipoint> &ipts,
                const int octaves = OCTAVES,
                const int intervals = INTERVALS,
                const int init_sample = INIT_SAMPLE,
                const float thres = THRES);

    //! Constructor with image
    FastHessian(cudaImage *d_img,
                std::vector<Ipoint> &ipts,
                const int octaves = OCTAVES,
                const int intervals = INTERVALS,
                const int init_sample = INIT_SAMPLE,
                const float thres = THRES);

    //! Save the parameters
    void saveParameters(const int octaves,
                        const int intervals,
                        const int init_sample,
                        const float thres);

    //! Set or re-set the integral image source
    void setIntImage(cudaImage *d_img);

    //! Find the image features and write into vector of features
    void getIpoints();

  private:
    //! Pointer to the integral Image, and its attributes
	cudaImage *d_img;
	int i_width, i_height;

	//! GPU pointer to keypoints
	fasthessian_cudaIpoint *d_points;

	//! Pointer to keypoints on host
	fasthessian_cudaIpoint *h_points;

    //! Reference to vector of features passed from outside
    std::vector<Ipoint> &ipts;

    //! Number of Octaves
    int octaves;

	//! Adjusted number of octaves depending on image size
	int adjd_octaves;

    //! Number of Intervals per octave
    int intervals;

    //! Initial sampling step for Ipoint detection
    int init_sample;

    //! Threshold value for blob resonses
    float thres;

	//! Array stack of determinant of hessian values
	float *d_det;
	size_t det_pitch;

    //! Calculate determinant of hessian responses
    void buildDet();

	//! Copy GPU keypoints to ipts ignoring zero entries
	void collectPoints(fasthessian_cudaIpoint *points);

	//! Intialize GPU
	void initGPU();
};

//--S [] 2013/03/06: Sang-Wook Lee
}  // namespace surfgpu
//--E [] 2013/03/06: Sang-Wook Lee

#endif /* FASTHESSIAN_H */
