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

#ifndef _FILESUSTEM_AUX_H_
#define _FILESUSTEM_AUX_H_

#include <QString>

namespace filesys {

  bool check_file(QString qsFilename);

  bool check_dir(QString qsDir);
  bool create_dir(QString qsDir);
  void split_filename(QString qsFilename, QString &qsPath, QString &qsName);
  void split_filename_ext(QString qsFilename, QString &qsPath, QString &qsBaseName, QString &qsExt);

  inline QString get_basename(QString qsFilename) {
    QString qsPath, qsBaseName, qsExt;
    split_filename_ext(qsFilename, qsPath, qsBaseName, qsExt);
    return qsBaseName;
  }

  void make_absolute(QString &qsFilename);


  QString add_suffix(QString qsFilename, QString qsSuffix);
}

#endif
