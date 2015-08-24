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

#ifndef _OBJECTDETECT_AUX_HPP_
#define _OBJECTDETECT_AUX_HPP_

namespace object_detect {


  /** 
      map negative classifier scores to small positive values

      Assume that classifier was not evaluated at locations with score == 0.

      It is important that scores are computed on sparse grid, otherwise marginals at different scales 
      will become uncomparable (spatial uncertainty grows with scale and we currently use unnormalized Gaussians
      for marginalization).

      Setting the minimum score to small positive value slightly improves the results over simply setting 
      it to 0.
  */
  template <typename Array>
  void clip_scores_fill(Array &grid, double min_val = 0.0001) {
    
    uint nElements = grid.num_elements();
    typename Array::element *pData = grid.data();

    int set_count = 0;

    for (uint i3 = 0; i3 < nElements; ++i3) {
      if (pData[i3] < 0) {
        pData[i3] = min_val;
        ++set_count;
      }
    }     
  }


}// namespace

#endif
