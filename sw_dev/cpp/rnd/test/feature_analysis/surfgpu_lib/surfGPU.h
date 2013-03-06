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

#ifndef SURF_H
#define SURF_H

#include <vector>

//--S [] 2013/03/06: Sang-Wook Lee
//#include "ipoint.h"
#include "ipointGPU.h"
//--E [] 2013/03/06: Sang-Wook Lee
#include "surf_cudaipoint.h"
#include "cudaimage.h"

//--S [] 2013/03/06: Sang-Wook Lee
namespace surfgpu {
//--E [] 2013/03/06: Sang-Wook Lee

class Surf {

  public:

    //! Standard Constructor (img is an integral image)
    Surf(cudaImage *img, IpVec &ipts);

	//! Standard Destructor
	~Surf();

    //! Describe all features in the supplied vector
    void getDescriptors(bool upright = false);

    //! Set or re-set the integral image source
    void setIntImage(cudaImage *d_img);

	//! Set or re-set the interest point source
	void setIpoints(IpVec &ipts);

  private:

    //---------------- Private Functions -----------------//
	
	//! Free integral image
 	void freeIntImage();

	void computeDescriptors(int upright);

    //---------------- Private Variables -----------------//

    //! Integral image where Ipoints have been detected
    cudaImage *d_img;
	int i_width, i_height;

    //! Ipoints vector
    IpVec &ipts;

	//! CUDA array holding the integral image
	cudaArray *ca_intimg;

	//! GPU memory for padded integral image
	float *d_intimg_padded;

	//! Device memory containing interest points
	surf_cudaIpoint *d_ipoints;
};

//--S [] 2013/03/06: Sang-Wook Lee
}  // namespace surfgpu
//--E [] 2013/03/06: Sang-Wook Lee

#endif /* SURF_H */
