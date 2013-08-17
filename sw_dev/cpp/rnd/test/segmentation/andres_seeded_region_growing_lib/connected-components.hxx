// Connected component labeling in n-dimensional grid graphs.
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
#ifndef ANDRES_VISION_CONNECTED_COMPONENTS_HXX
#define ANDRES_VISION_CONNECTED_COMPONENTS_HXX

#include <vector>
#include <queue>

#include "marray.hxx"

namespace andres {
namespace vision {

/// Connected component labeling in an n-dimension array, using the 2n-neighborhood
///
/// \param inputLabeling Array in which 0 is taken to be background label; 
///        values unequal to zero are taken to be connected components.
/// \param outputLabeling Array in which connected components are labeled, 
///        starting with the label 1 (background is labeled as 0).
/// \param sizes sizes of connected components
///
template<class T, class U, class V>
void connectedComponentLabeling(
    const View<T>& inputLabeling,
    View<U>& outputLabeling,
    std::vector<V>& sizes
) {
    if(inputLabeling.dimension() != outputLabeling.dimension()) {
        throw std::runtime_error("dimension of inputLabeling and componenets mismatch.");
    }
    for(size_t j = 0; j < inputLabeling.dimension(); ++j) {
        if(inputLabeling.shape(j) != outputLabeling.shape(j)) {
            throw std::runtime_error("shape of inputLabeling and outputLabeling mismatch.");
        }
    }

    // clear outputLabeling
    for(size_t j = 0; j < inputLabeling.size(); ++j) {
        outputLabeling(j) = static_cast<U>(0);
    }

    U currentLabel = 0;
    std::queue<size_t> queue;
    sizes.clear();
    std::vector<size_t> coordinate(inputLabeling.dimension());
    for(size_t j = 0; j < inputLabeling.size(); ++j) {
        if(inputLabeling(j) != static_cast<T>(0) && outputLabeling(j) == static_cast<U>(0)) {
            T currentinputLabelingValue = inputLabeling(j);
            currentLabel++;
            outputLabeling(j) = currentLabel; // label 
            sizes.push_back(static_cast<V>(1)); // set counter of current cc to 1
            queue.push(j);
            while(!queue.empty()) {
                size_t k = queue.front();
                queue.pop();
                inputLabeling.indexToCoordinates(k, coordinate.begin());
                for(unsigned short d = 0; d < inputLabeling.dimension(); ++d) {
                    if(coordinate[d] != 0) {
                        --coordinate[d];
                        if(inputLabeling(coordinate.begin()) == currentinputLabelingValue && outputLabeling(coordinate.begin()) == 0) {
                            size_t m;
                            inputLabeling.coordinatesToIndex(coordinate.begin(), m);
                            outputLabeling(m) = currentLabel; // label 
                            sizes[currentLabel - 1]++;
                            queue.push(m);
                        }
                        ++coordinate[d];
                    }
                }
                for(unsigned short d = 0; d < inputLabeling.dimension(); ++d) {
                    if(coordinate[d] < inputLabeling.shape(d) - 1) {
                        ++coordinate[d];
                        if(inputLabeling(coordinate.begin()) == currentinputLabelingValue && outputLabeling(coordinate.begin()) == 0) {
                            size_t m;
                            inputLabeling.coordinatesToIndex(coordinate.begin(), m);
                            outputLabeling(m) = currentLabel; // label 
                            sizes[currentLabel - 1]++;
                            queue.push(m);
                        }
                        --coordinate[d];
                    }
                }
            }
        }
    }
}

} // namespace vision
} // namespace andres

#endif // #ifndef ANDRES_VISION_CONNECTED_COMPONENTS_HXX
