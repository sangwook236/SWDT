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

#ifndef _MISC_HPP_
#define _MISC_HPP_

#include <QString>

/**
   various helper functions 
*/

template <class T>
inline T square(const T &x) { return x*x; };

/* test that the value is in [min, max) */
template<class T>
inline bool check_bounds(T val, T min, T max) 
{
  return (val >= min && val < max);
}

template <class T> 
inline void check_bounds_and_update(T &val, const T &min, const T &max)
{
  assert(min <= max - 1);

  if (val < min)
    val = min;

  if (val >= max)
    val = max - 1;
}

inline QString padZeros(QString qsStr, int npad)
{
  QString qsRes = qsStr;

  if (qsRes.length() < npad) 
    qsRes = QString(npad - qsRes.length(), '0') + qsRes;

  return qsRes;
}

#endif
