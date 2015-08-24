/** 
    This file is part of the implementation of the people detection and pose estimation model as described in the paper:
    
    M. Andriluka, S. Roth, B. Schiele. 
    Pictorial Structures Revisited: People Detection and Articulated Pose Estimation. 
    IEEE Conference on Computer Vision and Pattern Recognition (CVPR'09), Miami, USA, June 2009

    Please cite the paper if you are using this code in your work.

    The code may be used free of charge for non-commercial and
    educational purposes, the only requirement is that this text is
    preserved within the derivative work. For any other purpose you
    must contact the authors for permission. This code may not be
    redistributed without permission from the authors.  

    Author: Micha Andriluka, 2009
	    andriluka@cs.tu-darmstadt.de
	    http://www.mis.informatik.tu-darmstadt.de/People/micha
*/

#ifndef _MULTI_ARRAY_FILTER_HPP_
#define _MULTI_ARRAY_FILTER_HPP_

#include <libBoostMath/boost_math.h>

#include <libBoostMath/homogeneous_coord.h>
#include <libMultiArray/multi_array_transform.hpp>

namespace multi_array_op 
{
  using boost_math::double_matrix;
  using boost_math::double_vector;
  using boost::multi_array_types::index_range;

  template <typename Array1, typename Array2> 
  void grid_filter_1d(const Array1 &grid_in, Array2 &grid_out, const double_vector &f)
  {
    assert(Array1::dimensionality == 1 && Array2::dimensionality == 1);
    assert(grid_in.shape()[0] == grid_out.shape()[0]);

    int grid_size = grid_in.shape()[0];
    int ksize = (f.size() - 1)/2;

    for (int s = 0; s < grid_size; ++s) {
      int n1 = std::max(s - ksize, 0);
      int n2 = std::min(s + ksize, grid_size-1);
      float val = 0.0;
      for (int s1 = n1; s1 <= n2; ++s1)
        val += grid_in[s1]*f[s1 - (s - ksize)];

      grid_out[s] = val;
    }
  }

  /**
     apply isotropic Gaussian filter to 2d array 

     bNormalize true if the filter should be normalized
  */
  template <typename Array1, typename Array2>
  void gaussFilterDiag2d(const Array1 &grid_in, Array2 &grid_out, const double_matrix &C, bool bNormalize) {

    assert(C.size1() == 2 && C.size2() == 2);
    assert(C(0, 1) == 0 && C(1, 0) == 0);
    assert(C(0, 0) > 0 && C(1, 1) > 0);

    assert(Array1::dimensionality == 2);
    assert(Array2::dimensionality == 2);

    int grid_width = grid_in.shape()[1];
    int grid_height = grid_in.shape()[0];

    // it is not possible to resize grid_out if it is a view, enforce correct size
    assert((int)grid_out.shape()[0] == grid_height);
    assert((int)grid_out.shape()[1] == grid_width);

    double sigma_x = sqrt(C(0, 0));
    double sigma_y = sqrt(C(1, 1));

    double_vector f_x, f_y;
    boost_math::get_gaussian_filter(f_x, sigma_x, bNormalize);
    boost_math::get_gaussian_filter(f_y, sigma_y, bNormalize);

    FloatGrid2 grid_smooth_x(boost::extents[grid_height][grid_width]);

    for (int iy = 0; iy < grid_height; ++iy) {
      ConstFloatGrid2View1 view_in = grid_in[boost::indices[iy][index_range()]];
      FloatGrid2View1 view_out = grid_smooth_x[boost::indices[iy][index_range()]];
      multi_array_op::grid_filter_1d(view_in, view_out, f_x);
    }

    for (int ix = 0; ix < grid_width; ++ix) {
      FloatGrid2View1 view_in = grid_smooth_x[boost::indices[index_range()][ix]];
      FloatGrid2View1 view_out = grid_out[boost::indices[index_range()][ix]];
      multi_array_op::grid_filter_1d(view_in, view_out, f_y);
    }
  }


  /**
     apply Gaussian filter with arbitrary covariance matrix to 2d array

     - transform array to CS in which covariance matrix is diagonal (use bilinear interpolation)
     - apply isotropic filter
     - transform back

     bIsSparse: if true the input array is sparce (bilinear interpolation is replaced with direct mapping)
     offset: offset the grid simultaneously with the back transformation 
  */
  template <typename Array1, typename Array2>
  void gaussFilter2dOffset(const Array1 &grid_in, Array2 &grid_out, const double_matrix &C, 
                           const double_vector &offset, bool bNormalize, bool bIsSparse) {

    const int PADDING_VALUE = 0;

    assert(C.size1() == 2 && C.size2() == 2);
    assert(offset.size() == 2);
 
    // it is not possible to resize grid_out if it is a view, enforce correct size
    assert(grid_out.shape()[0] == grid_in.shape()[0]);
    assert(grid_out.shape()[1] == grid_in.shape()[1]);

    double_matrix V(2, 2);
    double_matrix E(2, 2);

    /* K = V*E*V', invK = V*inv(E)V' */
    boost_math::eig2d(C, V, E);
    double_matrix T21 = hc::get_homogeneous_matrix(trans(V), 0, 0);
    double_matrix T23;
    FloatGrid2 grid_in_transformed;

    if (bIsSparse) 
      multi_array_op::transform_grid_resize(grid_in, grid_in_transformed, T21, T23, PADDING_VALUE, TM_DIRECT);
    else
      multi_array_op::transform_grid_resize(grid_in, grid_in_transformed, T21, T23, PADDING_VALUE, TM_BILINEAR);

    FloatGrid2 grid_in_transformed_smooth(boost::extents[grid_in_transformed.shape()[0]][grid_in_transformed.shape()[1]]);

    gaussFilterDiag2d(grid_in_transformed, grid_in_transformed_smooth, E, bNormalize);
    double_matrix T42 = hc::get_homogeneous_matrix(V, -offset(0), -offset(1));
    double_matrix T43 = prod(T42, T23);

    multi_array_op::transform_grid_fixed_size(grid_in_transformed_smooth, grid_out,
                                              T43, PADDING_VALUE, TM_BILINEAR);    
  }

  /**
     convinience function (offset = 0)
  */
  template <typename Array1, typename Array2>
  void gaussFilter2d(const Array1 &grid_in, Array2 &grid_out, const double_matrix &C, 
                     bool bNormalize, bool bIsSparse) {
    assert(C.size1() == 2 && C.size2() == 2);
    
    bool bIsDiag = (C(0,1) == 0 && C(1,0) == 0);

    if (bIsDiag) {
      gaussFilterDiag2d(grid_in, grid_out, C, bNormalize);
    }
    else {
      double_vector offset = boost_math::double_zero_vector(2);
      gaussFilter2dOffset(grid_in, grid_out, C, offset, bNormalize, bIsSparse);
    }
  }
  


}// namespace 

#endif
