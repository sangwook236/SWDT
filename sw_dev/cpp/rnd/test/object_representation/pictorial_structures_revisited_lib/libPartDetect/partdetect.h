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

#ifndef _PART_DETECT_H_
#define _PART_DETECT_H_

#include <vector>

// boost::random (needed to generate random rectangles for negative set)
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/uniform_real.hpp>

#include <libAnnotation/annotationlist.h>

#include <libPartDetect/PartConfig.pb.h>
#include <libPartDetect/AbcDetectorParam.pb.h>
#include <libPartDetect/PartWindowParam.pb.h>

#include <libPartDetect/partdef.h>
#include <libPartDetect/FeatureGrid.h>

#include <libPartApp/partapp.h>

#include <libKMA2/kmaimagecontent.h>

namespace part_detect {

  const int NO_CLASS_VALUE = 0;

  /****************************************
  
   partdetect_test.cpp

  ****************************************/
  
  /**
     this function is used for bootstrapping and in partdetect
  */
  void partdetect_dense(const ExpParam &exp_param, const AbcDetectorParam &abc_param, 
			const PartWindowParam &window_param, 
			const AnnotationList &test_annolist, std::vector<AdaBoostClassifier> &v_abc,
			QString qsScoreGridDir, 
			int firstidx, int lastidx, bool flip, 
			bool bSaveImageScoreGrid, bool bAddImageBorder);

  void partdetect(PartApp &part_app, int firstidx, int lastidx, bool flip, bool bSaveImageScoreGrid);

  void computeScoreGrid(const AbcDetectorParam &abc_param, const PartWindowParam::PartParam &part_window_param,
			const AdaBoostClassifier &abc, double grid_step, double part_scale,  
			FeatureGrid &feature_grid, ScoreGrid &score_grid, bool bSqueeze = true);

  void squeezeScoreGrid(ScoreGrid &score_grid);


  /****************************************
  
   partdetect_train.cpp

  ****************************************/

  bool is_point_in_rect(PartBBox bbox, double point_x, double point_y);

  void bootstrap_get_rects(const PartApp &part_app, int imgidx, int pidx, 
			   int num_rects, double min_score, 
			   std::vector<PartBBox> &rects, std::vector<double> &rects_scale,
			   bool bIgnorePartRects, bool bDrawRects);

  void bootstrap_partdetect(PartApp &part_app, int firstidx, int lastidx);

  void prepare_bootstrap_dataset(const PartApp &part_app, const AnnotationList &annolist, 
				 int firstidx, int lastidx);


  /**
     train AdaBoost part classifier
  */
  void abc_train_class(PartApp &part_app, int pidx, bool bBootstrap);


  /**
     compute features for part bounding box
  */
  bool compute_part_bbox_features(const AbcDetectorParam &abcparam, const PartWindowParam &window_param, 
				  const PartBBox &bbox, 
				  kma::ImageContent *input_image, int pidx, std::vector<float> &all_features, 
				  PartBBox &adjusted_rect);


  bool compute_part_bbox_features_scale(const AbcDetectorParam &abcparam, const PartWindowParam &window_param, 
					const PartBBox &bbox, 
					kma::ImageContent *input_image, int pidx, double scale, 
					std::vector<float> &all_features,                                 
					PartBBox &adjusted_rect);

  /****************************************
  
   partdetect_aux.cpp

  ****************************************/

  void computeDescriptorGrid(kma::ImageContent *input_image, FeatureGrid &grid, QString qsDESC);

  int get_app_group_idx(const PartConfig &part_conf, int pid);

  int pidx_from_pid(const PartConfig &part_conf, int pid);


  /** 
      determine how many features fit into detection window for given object part     
      (number of features depends of window size and distance between features)
  */
  void get_window_feature_counts(const AbcDetectorParam &abc_param, const PartWindowParam::PartParam &part_window_param, 
				 int &grid_x_count, int &grid_y_count);

  /**
     compute average size of part bounding box
  */
  void compute_part_window_param(const AnnotationList &annolist, const PartConfig &partconf, PartWindowParam &windowparam);

  void sample_random_partrect(boost::variate_generator<boost::mt19937, boost::uniform_real<> > &gen_01,
			      double min_scale, double max_scale,
			      double min_rot, double max_rot, 
			      double min_pos_x, double max_pos_x,
			      double min_pos_y, double max_pos_y,
			      int rect_width, int rect_height, 
			      PartBBox &partrect, double &scale, double &rot);

}// namespace 

#endif 
