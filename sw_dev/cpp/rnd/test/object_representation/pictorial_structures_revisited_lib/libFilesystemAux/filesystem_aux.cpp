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

#include <QDir>
#include <QFileInfo>

#include <cassert>

#include "filesystem_aux.h"

namespace filesys {
  bool check_file(QString qsFilename) 
  {
    QFileInfo fi(qsFilename);
    return fi.exists() && fi.isFile();
  }

  bool check_dir(QString qsDir)
  {
    assert(!qsDir.isEmpty());
    QDir dir(qsDir);
    return dir.exists();
  }

  bool create_dir(QString qsDir) 
  {
    QFileInfo fi(qsDir);
  
    /* absolute path is necessary since mkpath works weird otherwise (2 dirs are created instead of one) */
    if (fi.isRelative())
      qsDir = fi.absoluteFilePath();

    if (check_dir(qsDir))
      return true;

    QDir dir(qsDir);
    //cout << "creating " << qsDir.toStdString() << endl;
    return dir.mkpath(qsDir);

  }

  void split_filename(QString qsFilename, QString &qsPath, QString &qsName)
  {
    QFileInfo fi(qsFilename);

    qsPath = fi.path();
    qsName = fi.fileName();
  }

  /**
     split full path into components

     qsPath does not end with '/'
     qsExt does not start with '.'
   */

  void split_filename_ext(QString qsFilename, QString &qsPath, QString &qsBaseName, QString &qsExt)
  {
    QFileInfo fi(qsFilename);

    qsPath = fi.path();
    qsBaseName = fi.baseName();
    qsExt = fi.completeSuffix();
  }

  QString add_suffix(QString qsFilename, QString qsSuffix)
  {
    QString qsPath, qsBaseName, qsExt;
    split_filename_ext(qsFilename, qsPath, qsBaseName, qsExt);
    return qsPath + "/" + qsBaseName + qsSuffix + "." + qsExt;
  }

  void make_absolute(QString &qsFilename) {
    QFileInfo fi(qsFilename);
    if (fi.isRelative())
      qsFilename = fi.absoluteFilePath();
  }


}
