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

#ifndef _MULTI_ARRAY_DEF_H_
#define _MULTI_ARRAY_DEF_H_

#include <boost/multi_array.hpp>

typedef boost::multi_array<float, 1> FloatGrid1;
typedef boost::multi_array<float, 2> FloatGrid2;

typedef boost::multi_array<double, 2> DoubleGrid2;

typedef boost::multi_array<long double, 2> LongDoubleGrid2;

typedef boost::multi_array<float, 3> FloatGrid3;
typedef boost::multi_array<float, 4> FloatGrid4;
typedef boost::multi_array<float, 5> FloatGrid5;

typedef boost::array_view_gen<FloatGrid2, 1>::type FloatGrid2View1;

typedef boost::array_view_gen<FloatGrid3, 1>::type FloatGrid3View1;
typedef boost::array_view_gen<FloatGrid3, 2>::type FloatGrid3View2;

typedef boost::array_view_gen<FloatGrid4, 2>::type FloatGrid4View2;

typedef boost::array_view_gen<FloatGrid5, 3>::type FloatGrid5View3;

typedef boost::const_array_view_gen<const FloatGrid2, 1>::type ConstFloatGrid2View1;


#endif
