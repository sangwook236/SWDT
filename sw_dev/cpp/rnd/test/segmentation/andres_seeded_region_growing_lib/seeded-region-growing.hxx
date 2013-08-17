// Seeded region growing in n-dimensional grid graphs, in linear time.
//
// Copyright (c) 2013 by Bjoern Andres.
// 
// This software was developed by Bjoern Andres.
// Enquiries shall be directed to bjoern@andres.sc.
//
// All advertising materials mentioning features or use of this software must
// display the following acknowledgement: ``This product includes andres::vision 
// developed by Bjoern Andres. Please direct enquiries concerning andres::vision 
// to bjoern@andres.sc''.
//
// Redistribution and use in source and binary forms, with or without 
// modification, are permitted provided that the following conditions are met:
//
// - Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// - Redistributions in binary form must reproduce the above copyright notice, 
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// - All advertising materials mentioning features or use of this software must 
//   display the following acknowledgement: ``This product includes 
//   andres::vision developed by Bjoern Andres. Please direct enquiries 
//   concerning andres::vision to bjoern@andres.sc''.
// - The name of the author must not be used to endorse or promote products 
//   derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR IMPLIED 
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF 
// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO 
// EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; 
// OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF 
// ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// 
#pragma once
#ifndef ANDRES_VISION_SEEDED_REGION_GROWING
#define ANDRES_VISION_SEEDED_REGION_GROWING

#include <vector>
#include <queue>
#include <algorithm> // max
#include <map>

#include "marray.hxx"
#include "connected-components.hxx"

namespace andres {
namespace vision {

namespace detail {
   template<class T> 
      inline bool isAtSeedBorder(const View<T>& seeds, const size_t index);
}

/// Seeded region growing in an n-dimension array, using the 2n-neighborhood.
///
/// \param elevation Elevation map (will be discretized to unsigned char)
/// \param segmentation Labeled grown regions
/// \param maxLevelSeeds Values smaller than or equal to this threshold become seeds
///
template<class T, class U>
void 
seededRegionGrowing(
    const View<T>& elevation,
    const unsigned char maxLevelSeeds,
    View<U>& segmentation // output
) {
    if(elevation.dimension() != segmentation.dimension()) {
        throw std::runtime_error("dimension of elevation and segmentation mismatch.");
    }
    for(size_t j = 0; j < elevation.dimension(); ++j) {
        if(elevation.shape(j) != segmentation.shape(j)) {
            throw std::runtime_error("shape of elevation and segmentation mismatch.");
        }
    }

    // compute discrete elevation 
    Marray<double> scratch(elevation); 
    double maxValue = 0.0;
    for(size_t j = 0; j < scratch.size(); ++j) {
        if(scratch(j) > maxValue) {
            maxValue = scratch(j);
        }
    }
    scratch *= (255.0 / maxValue);
    Marray<unsigned char> discreteElevation(scratch); // floor function and cast

    // compute seeds
    Marray<unsigned char> seeds(discreteElevation.shapeBegin(), discreteElevation.shapeEnd());
    for(size_t j = 0; j < discreteElevation.size(); ++j) {
        if(discreteElevation(j) <= maxLevelSeeds) {
            seeds(j) = 1;
        }
    }
    std::vector<size_t> sizes;
    connectedComponentLabeling(seeds, segmentation, sizes);
    
    seededRegionGrowing(discreteElevation, segmentation);
}

/// Seeded region growing in an n-dimension array, using the 2n-neighborhood
///
/// This function operates in-place on its second parameter.
///
/// \param elevation 8-bit Elevation map 
/// \param seeds As input: labeled seeds; as output: labeled grown regions
///
template<class T>
void 
seededRegionGrowing(
    const View<unsigned char>& elevation,
    View<T>& seeds
) {
    if(elevation.dimension() != seeds.dimension()) {
        throw std::runtime_error("dimension of elevation and seeds mismatch.");
    }
    for(size_t j = 0; j < elevation.dimension(); ++j) {
        if(elevation.shape(j) != seeds.shape(j)) {
            throw std::runtime_error("shape of elevation and seeds mismatch.");
        }
    }

    // define 256 queues, one for each gray level.
    std::vector<std::queue<size_t> > queues(256);

    // add each unlabeled pixels which is adjacent to a seed
    // to the queue corresponding to its gray level
    for(size_t j = 0; j < seeds.size(); ++j) {
        if(detail::isAtSeedBorder<T>(seeds, j)) {
            queues[elevation(j)].push(j);
        }
    }

    // grow
    unsigned char grayLevel = 0;
    std::vector<size_t> coordinate(elevation.dimension());
    for(;;) {
        while(!queues[grayLevel].empty()) {
            // label pixel and remove from queue
            size_t j = queues[grayLevel].front();
            queues[grayLevel].pop();

            // add unlabeled neighbors to queues
            seeds.indexToCoordinates(j, coordinate.begin());
            for(unsigned char d = 0; d < elevation.dimension(); ++d) {
                if(coordinate[d] != 0) {
                    --coordinate[d];
                    if(seeds(coordinate.begin()) == 0) {
                        size_t k;
                        seeds.coordinatesToIndex(coordinate.begin(), k);
                        unsigned char queueIndex = std::max(elevation(coordinate.begin()), grayLevel);
                        seeds(k) = seeds(j); // label pixel
                        queues[queueIndex].push(k);
                    }
                    ++coordinate[d];
                }
            }
            for(unsigned char d = 0; d < elevation.dimension(); ++d) {
                if(coordinate[d] < seeds.shape(d) - 1) {
                    ++coordinate[d];
                    if(seeds(coordinate.begin()) == 0) {
                        size_t k;
                        seeds.coordinatesToIndex(coordinate.begin(), k);
                        unsigned char queueIndex = std::max(elevation(coordinate.begin()), grayLevel);
                        seeds(k) = seeds(j); // label pixel
                        queues[queueIndex].push(k);
                    }
                    --coordinate[d];
                }
            }
        }
        if(grayLevel == 255) {
            break;
        }
        else {
            queues[grayLevel] = std::queue<size_t>(); // free memory
            ++grayLevel;
        }
    }
}

// \cond SUPPRESS_DOXYGEN
namespace detail {

template<class T>
inline bool isAtSeedBorder(
    const View<T>& seeds,
    const size_t index
) {
    if(seeds(index) == 0) {	
        return false; // not a seed voxel
    }
    else {
        std::vector<size_t> coordinate(seeds.dimension());
        seeds.indexToCoordinates(index, coordinate.begin());
        for(unsigned char d = 0; d < seeds.dimension(); ++d) {
            if(coordinate[d] != 0) {
                --coordinate[d];
                if(seeds(coordinate.begin()) == 0) {
                    return true;
                }
                ++coordinate[d];
            }
        }
        for(unsigned char d = 0; d < seeds.dimension(); ++d) {
            if(coordinate[d] < seeds.shape(d) - 1) {
                ++coordinate[d];
                if(seeds(coordinate.begin()) == 0) {
                    return true;
                }
                --coordinate[d];
            }
        }
        return false;
    }
}

} // namespace detail
// \endcond 

} // namespace vision
} // namespace andres

#endif // #ifndef ANDRES_VISION_SEEDED_REGION_GROWING
