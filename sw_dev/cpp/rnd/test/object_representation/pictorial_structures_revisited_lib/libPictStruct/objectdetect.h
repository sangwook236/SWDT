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

#ifndef _OBJECT_DETECT_H_
#define _OBJECT_DETECT_H_

#include <vector>

#include <libAnnotation/annotationlist.h>

#include <libPartApp/partapp.h>

#include <libPartDetect/PartConfig.pb.h>

#include <libBoostMath/boost_math.h>

#include <libMultiArray/multi_array_def.h>

#include <libPictStruct/HypothesisList.pb.h>

namespace object_detect {

  enum ScoreProbMapType {
    SPMT_NONE = 0,
    SPMT_RATIO = 1,
    SPMT_RATIO_WITH_PRIOR = 2,
    SPMT_SPATIAL = 3,
    SPMT_SPATIAL_PROD = 4
  };

  typedef HypothesisList::ObjectHypothesis ObjectHypothesis;

  int JointTypeIdFromString(QString qsType);

  struct Joint {
    enum {POS_GAUSSIAN = 1, ROT_GAUSSIAN = 2};

    Joint():rot_mean(0), rot_sigma(0) {}

    Joint(int _type, int _child_idx, int _parent_idx, 
	  boost_math::double_vector _offset_c, 
	  boost_math::double_vector _offset_p, 
	  boost_math::double_matrix _C);

    Joint(int _type, int _child_idx, int _parent_idx, 
	  boost_math::double_vector _offset_c, 
	  boost_math::double_vector _offset_p, 
	  boost_math::double_matrix _C, 
	  double _rot_mean, 
	  double _rot_sigma);

    int type;

    int child_idx; // id of child part (not index, i.e. this is indepent of the order in which parts are stored)
    int parent_idx; // id of parent part

    boost_math::double_vector offset_c; // parent_pos - child_pos (in pixels)
    boost_math::double_vector offset_p; // child_pos - parent_pos (in pixels)
    boost_math::double_matrix C; 

    double rot_mean;
    double rot_sigma;
  };

  /****************************************
  
   objectdetect_learnparam.cpp

  ****************************************/
  void save_joint(const PartApp &part_app, Joint &joint);
  void load_joint(const PartApp &part_app, int jidx, Joint &joint);

  void learn_conf_param(const PartApp &part_app, const AnnotationList &train_annolist);

  /****************************************
  
   objectdetect_aux.cpp

  ****************************************/

  void loadJoints(const PartApp &part_app, std::vector<Joint> &joints, bool flip);

  void findLocalMax(const ExpParam &exp_param, const FloatGrid3 &log_prob_grid, 
                    HypothesisList &hypothesis_list,
                    int max_hypothesis_number);

  int JointTypeFromString(QString qsType);

  int MapTypeFromString(QString qsScoreProbMapType);

  QString MapTypeToString(int maptype) ;
    
  QString getObjectHypFilename(int imgidx, bool flip, int scoreProbMapType);

  void findObjectDataset(const PartApp &part_app, int firstidx, int lastidx, int scoreProbMapType);

  void nms_recursive(const HypothesisList hypothesis_list, 
                     std::vector<bool> &nms, 
                     double train_object_width, 
                     double train_object_height);

  void saveRecoResults(const PartApp &part_app, int scoreProbMapType);


  /****************************************
  
   objectdetect_findpos.cpp

  ****************************************/

   void findObjectImagePosJoints(const PartApp &part_app, int imgidx, bool flip, HypothesisList &hypothesis_list,  
 		       int scoreProbMapType); 

  /****************************************
  
   objectdetect_findrot.cpp

  ****************************************/

  void get_incoming_joints(const std::vector<Joint> &joints, int curidx, std::vector<int> &all_children, std::vector<int> &all_joints);

  void computeRotJointMarginal(const ExpParam &exp_param, 
                               FloatGrid3 &log_prob_child, FloatGrid3 &log_prob_parent, 
                               const boost_math::double_vector &_offset_c_10, const boost_math::double_vector &_offset_p_01, 
                               const boost_math::double_matrix &C, 
                               double rot_mean, double rot_sigma,
                               double scale, bool bIsSparse);

  void findObjectImageRotJoints(const PartApp &part_app, int imgidx, bool flip, HypothesisList &hypothesis_list, int scoreProbMapType);

}
#endif
