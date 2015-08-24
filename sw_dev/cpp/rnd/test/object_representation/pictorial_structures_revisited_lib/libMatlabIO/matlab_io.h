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

#ifndef _MATLAB_IO_H
#define _MATLAB_IO_H

/* MATLAB header file */
#include <mat.h>
//--S [] 2013/01/10: Sang-Wook Lee
//#include <mex.h>
//--E [] 2013/01/10

#include <QString>

#include <libBoostMath/boost_math.h>

namespace matlab_io {

  /**
     open .mat file, mode is "w" for writing and "r" for reading, "wz" for writing compressed data

     return false if file could not be opened
  */
  MATFile *mat_open(QString qsFilename, const char *mode);

  void mat_close(MATFile *f);

  /**
     save/load ublas matrices and vectors
   */

  bool mat_save_double_vector(QString qsFilename, QString qsVarName, const boost_math::double_vector &v);
  bool mat_save_double_matrix(QString qsFilename, QString qsVarName, const boost_math::double_matrix &m);

  bool mat_save_double_vector(MATFile *f, QString qsVarName, const boost_math::double_vector &v);
  bool mat_save_double_matrix(MATFile *f, QString qsVarName, const boost_math::double_matrix &m);

  bool mat_load_double_vector(QString qsFilename, QString qsVarName, boost_math::double_vector &v); 
  bool mat_load_double_matrix(QString qsFilename, QString qsVarName, boost_math::double_matrix &m);

  /** 
      save/load scalar doubles
   */
  bool mat_save_double(QString qsFilename, QString qsVarName, double d); 
  bool mat_save_double(MATFile *f, QString qsVarName, double d); 

  bool mat_load_double(QString qsFilename, QString qsVarName, double &d); 

  bool mat_save_std_vector(MATFile *f, QString qsVarName, const std::vector<float> &v); 


}// namespace 

#endif

