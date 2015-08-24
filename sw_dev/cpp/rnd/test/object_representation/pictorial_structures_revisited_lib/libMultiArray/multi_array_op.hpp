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

#ifndef _MULTI_ARRAY_OP_H_
#define _MULTI_ARRAY_OP_H_

#include <cassert>

#include <libMultiArray/multi_array_def.h>
#include <libBoostMath/boost_math.h>

#include <boost/lambda/lambda.hpp>

namespace multi_array_op 
{
  using std::cout;
  using std::endl;

  template <typename Array>
  void normalize(Array &a)
  {
    using namespace boost::lambda;

    typename Array::element *pData = a.data();
    uint nElements = a.num_elements();
 
    typename Array::element *pDataEnd = pData + nElements;
    typename Array::element val_sum = std::accumulate(pData, pDataEnd, 0.0);

    assert(val_sum > 0);

    std::for_each(pData, pDataEnd, _1 = _1 / val_sum);
  }

  template <typename Array> 
  void getMinMax(const Array &a, typename Array::element &minval, typename Array::element &maxval)
  {
    const typename Array::element *pData = a.data();
    uint nElements = a.num_elements();
  
    minval = std::numeric_limits<typename Array::element>::infinity();
    maxval = -std::numeric_limits<typename Array::element>::infinity();

    for (uint i = 0; i < nElements; ++i) {
      if (pData[i] > maxval) 
        maxval = pData[i];

      if (pData[i] < minval)
        minval = pData[i];
    }
    
  }

  template <typename Array>
  void multGrid(Array &grid, typename Array::element scalar)
  {
    typename Array::element *data = grid.data();
    int nElements = grid.num_elements();

    for (int i = 0; i < nElements; ++i) 
      data[i] = scalar*data[i];
  }

  template <typename Array> 
  void setGrid(Array &grid, typename Array::element num) {
    typename Array::element *pData = grid.data();
    int nElements = grid.num_elements();

    for (int i = 0; i < nElements; ++i)
      pData[i] = num;
  }

  template <typename Array> 
  void addGrid1(Array &grid, typename Array::element num)
  {
    typename Array::element *pData = grid.data();
    int nElements = grid.num_elements();

    for (int i = 0; i < nElements; ++i)
      pData[i] += num;
  }

  template <typename Array1, typename Array2> 
  void addGrid2(Array1 &grid1, const Array2 &grid2)
  {
    assert(grid1.storage_order() == grid2.storage_order());

    typename Array1::element *pData1 = grid1.data();
    const typename Array2::element *pData2 = grid2.data();

    assert(grid1.num_elements() == grid2.num_elements());

    int nElements = grid1.num_elements();
    for (int i = 0; i < nElements; ++i)
      pData1[i] += pData2[i];
  }

  template <typename Array1, typename Array2> 
  void multGrid2(Array1 &grid1, const Array2 &grid2)
  {
    assert(grid1.storage_order() == grid2.storage_order());

    typename Array1::element *pData1 = grid1.data();
    const typename Array2::element *pData2 = grid2.data();

    assert(grid1.num_elements() == grid2.num_elements());

    int nElements = grid1.num_elements();
    for (int i = 0; i < nElements; ++i)
      pData1[i] *= pData2[i];
  }

  template <typename Array1, typename Array2> 
  void divGrid2(Array1 &grid1, const Array2 &grid2)
  {
    assert(grid1.storage_order() == grid2.storage_order());

    typename Array1::element *pData1 = grid1.data();
    const typename Array2::element *pData2 = grid2.data();

    assert(grid1.num_elements() == grid2.num_elements());

    int nElements = grid1.num_elements();
    for (int i = 0; i < nElements; ++i)
      pData1[i] /= pData2[i];
  }

  template <typename Array>
  void computeLogGrid(Array &prob_grid)
  {
    typename Array::element *data = prob_grid.data();
    int nElements = prob_grid.num_elements();

    for (int i = 0; i < nElements; ++i) {
      assert(data[i] >= 0);

      if (data[i] == 0)
        data[i] = LOG_ZERO;
      else
        data[i] = log(data[i]);
    }
  }

  template <typename Array>
  void computeExpGrid(Array &prob_grid)
  {
    typename Array::element *data = prob_grid.data();
    int nElements = prob_grid.num_elements();

    for (int i = 0; i < nElements; ++i) {
      assert(data[i] < std::numeric_limits<double>::max());
      data[i] = exp(data[i]);
      assert(data[i] < std::numeric_limits<double>::max());
    }
  }

  template <typename Array>
  void computeNegLogGrid(Array &prob_grid)
  {
    typename Array::element *data = prob_grid.data();
    int nElements = prob_grid.num_elements();

    for (int i = 0; i < nElements; ++i) {
      assert(data[i] >= 0);

      if (data[i] == 0)
        data[i] = NEG_LOG_ZERO;
      else
        data[i] = -log(data[i]);
    }
  }

  template <typename Array> 
  void array_to_matrix(const Array &ar, boost_math::double_matrix &m)
  {
    int width = ar.shape()[1];
    int height = ar.shape()[0];

    m.resize(height, width);
    for (int ix = 0; ix < width; ++ix)
      for (int iy = 0; iy < height; ++iy)
        m(iy, ix) = ar[iy][ix];
  }

  template <typename Array>
  void clip_scores(Array &grid, float clip_val) {
    uint nElements = grid.num_elements();
    typename Array::element *pData = grid.data();

    for (uint i3 = 0; i3 < nElements; ++i3) {
      if (pData[i3] < clip_val)
        pData[i3] = clip_val;
    }     
  }

  /**
     this is supposed to be used for setting the classifier scores computed 
     at positions with fixed offset to some minimum value 

     note: it is assumed that unclassified positions have score = 0

     note: we only change scores at evaluated locations (i.e. score != 0),  
           setting scores at all positions to some min_val > 0 would make 
           marginals at different scales incomparable 
   */
  template <typename Array>
  void clip_scores_grid_minval(Array &grid, double min_val = 1e-4) {
    //double min_val = 0.0001;
    
    uint nElements = grid.num_elements();
    typename Array::element *pData = grid.data();

    //int set_count = 0;

    for (uint i3 = 0; i3 < nElements; ++i3) {
      if (pData[i3] < 0) {
        pData[i3] = min_val;
        //++set_count;
      }
    }     
    //cout << "clip_scores_test: minval: " << min_val << ", set_count: " << set_count << endl;
  }


}// namespace 



#endif
