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

#ifndef _PART_APP_H_
#define _PART_APP_H_

#include <QString>

#include <libAdaBoost/AdaBoost.h>
#include <libAnnotation/annotationlist.h>

#include <libMultiArray/multi_array_def.h>

#include <libPartDetect/AbcDetectorParam.pb.h>
#include <libPartDetect/PartConfig.pb.h>
#include <libPartDetect/PartWindowParam.pb.h>
#include <libPartDetect/partdef.h>

#include "ExpParam.pb.h"

/**
   this function is implemented in libPartApp since it is used in both parteval.cpp and scoreparams_test.cpp
   and i did not find a better place for it
 */

void bbox_from_pos(const ExpParam &exp_param, const PartWindowParam::PartParam &part_param, 
                   int scaleidx, int rotidx, int ix, int iy, 
                   PartBBox &bbox);


/** 
    PartApp is a container for options/datasets used in the experiment 
    (better name would be something like ExpDef)

    here we make a library out of it since it is used in 
      libPartDetect 
      libPictStruct
      apps/partapp

    all of which are potentially independent
*/

class PartApp {
 public:

  /* specify whether partapp is allowed to update the classifiers and joint parameters */
  bool m_bExternalClassDir;
  int m_rootpart_idx;

  ExpParam m_exp_param;
  PartConfig m_part_conf;
  AbcDetectorParam m_abc_param;
  PartWindowParam m_window_param;

  AnnotationList m_train_annolist;
  AnnotationList m_validation_annolist;
  AnnotationList m_test_annolist;

  PartApp() : m_bExternalClassDir(false), m_rootpart_idx(-1) {}

  void init(QString qsExpParam);

  QString getClassFilename(int pidx, int bootstrap_type) const;

  /* bootstrap_type: 
       0 - no bootstrapping 
       1 - bootstrapping 
       2 - first try to load classifier with bootstrapping, if unavailable load the classifier without bootstrapping
   */

  void loadClassifier(AdaBoostClassifier &abc, int pidx, int bootstrap_type = 2) const;
  void saveClassifier(AdaBoostClassifier &abc, int pidx, int bootstrap_type = 2) const;

  void getScoreGridFileName(int imgidx, int pidx, bool flip, 
			    QString &qsFilename, QString &qsVarName) const; 

  void loadScoreGrid(std::vector<std::vector<FloatGrid2> > &score_grid, int imgidx, int pidx,  
 		     bool flip, bool bInterpolate = true) const;

  void saveScoreGrid(std::vector<std::vector<FloatGrid2> > &score_grid, int imgidx, int pidx,  
		     bool flip) const; 

  FloatGrid3 loadPartMarginal(int imgidx, int pidx, int scaleidx, bool flip) const;

};


#endif
