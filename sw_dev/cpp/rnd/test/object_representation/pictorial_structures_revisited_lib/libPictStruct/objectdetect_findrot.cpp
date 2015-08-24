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


#include <libMultiArray/multi_array_op.hpp>
#include <libMultiArray/multi_array_transform.hpp>
#include <libMultiArray/multi_array_filter.hpp>

#include <libBoostMath/boost_math.h>
#include <libBoostMath/homogeneous_coord.h>

#include <libPartApp/partapp_aux.hpp>
#include <libProtoBuf/protobuf_aux.hpp>

#include <libPartDetect/partdetect.h>

#include <libMatlabIO/matlab_io.hpp>
#include <libMatlabIO/matlab_cell_io.hpp>

#include "objectdetect_aux.hpp"
#include "objectdetect.h"

using boost_math::double_vector;
using boost_math::double_matrix;
using boost_math::double_zero_matrix;

using boost::multi_array_types::index_range;

using namespace boost::lambda;
using namespace std;

namespace object_detect {

  void get_incoming_joints(const vector<Joint> &joints, int curidx, vector<int> &all_children, vector<int> &all_joints)
  {
    all_children.clear();
    all_joints.clear();

    for (uint jidx = 0; jidx < joints.size(); ++jidx) {
      if (joints[jidx].parent_idx == curidx) {
        all_children.push_back(joints[jidx].child_idx);
        all_joints.push_back(jidx);
      }
    }

  }

  /**
     compute part marginals (i.e. pass the messages from root downstream)

     assumptions:
     log_part_posterior contains product of messages from downstream and part appearance
     log_from_root product of messages received by root form all children excluding one particular child

     on completion:
     log_part_posterior contains marginal distribution of each part
     log_from_root contains upstream messages received by each part
   */

  void computePartMarginals(const PartApp &part_app, 
                           vector<Joint> joints,
                           int rootpart_idx, 
                           int scaleidx,
                           bool flip,
                           int imgidx, 
                           vector<vector<FloatGrid3> > &log_part_detections, 
                           vector<FloatGrid3> &log_part_posterior,
                           vector<FloatGrid3> &log_from_root)
  {

    QString qsPartMarginalsDir = (part_app.m_exp_param.log_dir() + "/" + 
                                  part_app.m_exp_param.log_subdir() + "/part_marginals").c_str();

    if (!filesys::check_dir(qsPartMarginalsDir))
      filesys::create_dir(qsPartMarginalsDir);

    int nParts = part_app.m_part_conf.part_size();
    int nRotations = part_app.m_exp_param.num_rotation_steps();
    double scale = scale_from_index(part_app.m_exp_param, scaleidx);

    int img_width = log_part_detections[0][0].shape()[2];
    int img_height = log_part_detections[0][0].shape()[1];

    assert(joints.size() > 0);
    assert((int)log_part_posterior.size() == nParts);
    assert((int)log_from_root.size() == nParts);

    vector<bool> vComputedRootMessages(nParts, false);
    vector<int> compute_marginals_stack;

    /** finish computation of messages from root to children */

    cout << "finalizing computation of upstream messages" << endl;
    vector<int> all_children;
    vector<int> incoming_joints;
    get_incoming_joints(joints, rootpart_idx, all_children, incoming_joints);
            
    for (int i = 0; i < (int)all_children.size(); ++i) {
      int child_idx = all_children[i];
      int jidx = incoming_joints[i];
      cout << "\tadding apprearance component to message to " << child_idx << endl;
      multi_array_op::addGrid2(log_from_root[child_idx], log_part_detections[rootpart_idx][scaleidx]);
              
      FloatGrid3 tmpgrid2(boost::extents[nRotations][img_height][img_width]);

      /** backward pass */
      computeRotJointMarginal(part_app.m_exp_param, 
                              log_from_root[child_idx], tmpgrid2, 
                              joints[jidx].offset_p, joints[jidx].offset_c, 
                              joints[jidx].C,
                              -joints[jidx].rot_mean, joints[jidx].rot_sigma, 
                              scale, false /* not sparse */ );

      log_from_root[child_idx] = tmpgrid2;
          
      vComputedRootMessages[child_idx] = true;
      compute_marginals_stack.push_back(child_idx);

    }// joints from root

    cout << "done." << endl;

    /* send downstream messages */

    while(!compute_marginals_stack.empty()) {
      int curidx = compute_marginals_stack.back();

      compute_marginals_stack.pop_back();
      cout << "computing marginal for part " << curidx << endl;

      assert(vComputedRootMessages[curidx] == true);

      /** this is part posterior (product of downstream messages, upstream message and part appearance ) */
      multi_array_op::addGrid2(log_part_posterior[curidx], log_from_root[curidx]);
                
      vector<int> all_children;
      vector<int> all_joints;
      get_incoming_joints(joints, curidx, all_children, all_joints);

      /** support only simple trees for now, otherwise we must make sure that children do not receive any messages which 
          they have send to the parent 
      */
      assert(all_children.size() <= 1); 

      /** compute messages to child nodes */
      if (all_children.size() == 1) {

        //for (int i = 0; i < (int)all_children.size(); ++i) {
        int i = 0;

        int child_idx = all_children[i];
        int jidx = all_joints[i];
          
        FloatGrid3 tmpgrid = log_part_detections[curidx][scaleidx];
        multi_array_op::addGrid2(tmpgrid, log_from_root[curidx]);

        computeRotJointMarginal(part_app.m_exp_param, 
                                tmpgrid, log_from_root[child_idx], 
                                joints[jidx].offset_p, joints[jidx].offset_c, 
                                joints[jidx].C, 
                                -joints[jidx].rot_mean, joints[jidx].rot_sigma, 
                                scale, false /* not sparse */); 

        vComputedRootMessages[child_idx] = true;
        compute_marginals_stack.push_back(child_idx);
      }// children


    }// children stack 


    /** save part posterior */
    for (int pidx = 0; pidx < nParts; ++pidx) {
      QString qsFilename = qsPartMarginalsDir + "/log_part_posterior_final" + 
        "_imgidx" + QString::number(imgidx) +
        "_scaleidx" + QString::number(scaleidx) +
        "_o" + QString::number((int)flip) + 
        "_pidx" + QString::number(pidx) + ".mat";

      cout << "saving " << qsFilename.toStdString() << endl;

      matlab_io::mat_save_multi_array(qsFilename, "log_prob_grid", log_part_posterior[pidx]);
    }
  }

  /**
     rot_mean, rot_sigma - mean and sigma of gaussian which describes range of possible rotations (both are in radians!!!)

   */
  void computeRotJointMarginal(const ExpParam &exp_param, 
                               FloatGrid3 &log_prob_child, FloatGrid3 &log_prob_parent, 
                               const double_vector &_offset_c_10, const double_vector &_offset_p_01, 
                               const double_matrix &C, 
                               double rot_mean, double rot_sigma,
                               double scale, bool bIsSparse)
  {
    cout << "computeRotJointMarginal, scale " << scale << endl;

    assert(rot_sigma > 0);
    assert(_offset_p_01.size() == 2 && _offset_c_10.size() == 2);
    double_vector offset_c_10 = hc::get_vector(_offset_c_10(0), _offset_c_10(1));
    double_vector offset_p_01 = hc::get_vector(_offset_p_01(0), _offset_p_01(1));
   
    assert(log_prob_child.shape()[0] == log_prob_parent.shape()[0]);
    assert(log_prob_child.shape()[1] == log_prob_parent.shape()[1]);
    assert(log_prob_child.shape()[2] == log_prob_parent.shape()[2]);

    int nRotations = log_prob_child.shape()[0];
    int grid_height = log_prob_child.shape()[1];
    int grid_width = log_prob_child.shape()[2];

    double rot_step_size = (exp_param.max_part_rotation() - exp_param.min_part_rotation())/exp_param.num_rotation_steps();
    rot_step_size *= M_PI / 180.0;

    assert(rot_step_size > 0);
    double rot_sigma_idx = rot_sigma / rot_step_size;

    /** rot_mean is a mean rotation from parent to child in world cs                                    */
    /** since we are propagating information from child to parent the mean rotation is multiplied by -1 */
    int rot_mean_idx = boost_math::round(-rot_mean / rot_step_size);

    cout << "-rot_mean: " << -rot_mean << ", rot_mean_idx: " << rot_mean_idx << endl;
    cout << "rot_sigma: " << rot_sigma << ", rot_sigma_idx: " << rot_sigma_idx << endl;

    /* transform from child part center to position of joint between child and parent parts */
    /*                                                                                      */
    /* position of joint depends on part scale and rotation                                 */
    /* joint 1-0: joint of child to parent                                                  */
    /* joint 0-1: joint of parent to child                                                  */
    /* ideally both have the same position in world coordinates                             */
    /* here we allow some offset which is penalized by euclidian distance                   */ 

    FloatGrid3 log_joint_01(boost::extents[nRotations][grid_height][grid_width]);
    FloatGrid3 log_joint_10(boost::extents[nRotations][grid_height][grid_width]);

    multi_array_op::setGrid(log_joint_10, LOG_ZERO);
    multi_array_op::setGrid(log_joint_01, LOG_ZERO);

    for (int rotidx = 0; rotidx < nRotations; ++rotidx) {
      int rotidx_out = rotidx + rot_mean_idx;

      if (rotidx_out >= 0 && rotidx_out < nRotations) {
        float alpha = rot_from_index(exp_param, rotidx)*M_PI/180.0;
        double_matrix Tgc = prod(hc::get_rotation_matrix(alpha), hc::get_scaling_matrix(scale));
        double_vector offset_g_10 = prod(Tgc, offset_c_10);

        double_matrix Tjoint = hc::get_translation_matrix(offset_g_10(0), offset_g_10(1));

        FloatGrid3View2 view_out = log_joint_10[boost::indices[rotidx_out][index_range()][index_range()]];

        multi_array_op::transform_grid_fixed_size(log_prob_child[rotidx], view_out, 
                                                  Tjoint, LOG_ZERO, TM_NEAREST);
      }
    }// rotations
      
    /* obtain probabilities */
    multi_array_op::computeExpGrid(log_joint_10); 

    /** compute values as if we would be rescaling every image to the training scale -> all normalization constants are 
        the same -> ignore them */

    bool bNormalizeRotFilter = false; 
    bool bNormalizeSpatialFilter = false;

    FloatGrid3 rot_filter_result(boost::extents[nRotations][grid_height][grid_width]);
    double_vector f_rot;
    boost_math::get_gaussian_filter(f_rot, rot_sigma_idx, bNormalizeRotFilter);
  
    /* rotation dimension */
    for (int x = 0; x < grid_width; ++x)
      for (int y = 0; y < grid_height; ++y) {
        FloatGrid3View1 view_in = log_joint_10[boost::indices[index_range()][y][x]];
        FloatGrid3View1 view_out = rot_filter_result[boost::indices[index_range()][y][x]];
        multi_array_op::grid_filter_1d(view_in, view_out, f_rot);
      }

    /* x/y dimensions */
    double_matrix scaleC = square(scale)*C;
    for (int rotidx = 0; rotidx < nRotations; ++rotidx) {
      FloatGrid3View2 view_in = rot_filter_result[boost::indices[rotidx][index_range()][index_range()]];
      FloatGrid3View2 view_out = log_joint_01[boost::indices[rotidx][index_range()][index_range()]];

      multi_array_op::gaussFilter2d(view_in, view_out, scaleC, bNormalizeSpatialFilter, bIsSparse);
    }

    /* go back to log prob */
    multi_array_op::computeLogGrid(log_joint_01);
      
    /* transform joint 0-1 obtain position of parent */
    for (int rotidx = 0; rotidx < nRotations; ++rotidx) {
      float alpha = rot_from_index(exp_param, rotidx)*M_PI/180.0;
      double_matrix Tgo = prod(hc::get_rotation_matrix(alpha), hc::get_scaling_matrix(scale));
      double_vector offset_g_01 = prod(Tgo, offset_p_01);

      double_matrix Tobject = hc::get_translation_matrix(-offset_g_01(0), -offset_g_01(1));
      FloatGrid3View2 view_out = log_prob_parent[boost::indices[rotidx][index_range()][index_range()]];

      multi_array_op::transform_grid_fixed_size(log_joint_01[rotidx], view_out, Tobject, 
                                                LOG_ZERO, TM_NEAREST);

    }
    
    cout << "done." << endl;
  }


  void computeRootPosteriorRot(const PartApp part_app, 
                               vector<vector<FloatGrid3> > &log_part_detections, 
                               FloatGrid3 &root_part_posterior, int rootpart_idx, vector<Joint> joints, bool flip, 
                               bool bIsSparse, int imgidx)
  {
    cout << "computeRootPosteriorRot" << endl;

    bool bDebugMessageOutput = false;

    /** this determines whether we also propagate messages from root back to the parts, this is not necessary
        if we only want to detect an object
     */
    bool bComputePartMarginals = part_app.m_exp_param.compute_part_marginals(); 

    /**
       number of samples to be drawn from posterior
     */
    int num_samples = part_app.m_exp_param.num_pose_samples();

    cout << "\t bComputePartMarginals: " << bComputePartMarginals << endl;
    cout << "\t num_samples: " << num_samples << endl;

    int nParts = part_app.m_part_conf.part_size();
    int nScales = part_app.m_exp_param.num_scale_steps();
    int nRotations = part_app.m_exp_param.num_rotation_steps();

    assert((int)log_part_detections.size() == nParts);
    assert((int)log_part_detections[0].size() == nScales);
    assert((int)log_part_detections[0][0].shape()[0] == nRotations);

    int img_width = log_part_detections[0][0].shape()[2];
    int img_height = log_part_detections[0][0].shape()[1];
    assert(img_width > 0 && img_height > 0);

    /** grid where we store results */
    FloatGrid4 root_part_posterior_full(boost::extents[nScales][nRotations][img_height][img_width]);
    multi_array_op::computeLogGrid(root_part_posterior_full);


    /** compute object probability for each scale */
    for (int scaleidx = 0; scaleidx < nScales; ++scaleidx) {

      /* 
         enforce upright orientation if necessary 
      */

      for (int pidx = 0; pidx < nParts; ++pidx) {
        if (part_app.m_part_conf.part(pidx).is_upright()) {

          for (int ridx = 0; ridx < nRotations; ++ridx) {
            double cur_rot = rot_from_index(part_app.m_exp_param, ridx);

            if (!(abs(cur_rot) < 15.0)) {
              for (int iy = 0; iy < img_height; ++iy)
                for (int ix = 0; ix < img_width; ++ix) {
                  log_part_detections[pidx][scaleidx][ridx][iy][ix] = LOG_ZERO;                  
                }
            }
          }
        }
      } // parts


      /* 
         strip border detections 
      */
      if (part_app.m_exp_param.strip_border_detections() > 0) {
        assert(part_app.m_exp_param.strip_border_detections() < 0.5);

        int strip_width = (int)(part_app.m_exp_param.strip_border_detections() * img_width);

        cout << "strip all detections of root part (" << 
          rootpart_idx << ") inside border region, strip_width: " << 
          strip_width << endl;

        for (int scaleidx = 0; scaleidx < nScales; ++scaleidx)
          for (int ridx = 0; ridx < nRotations; ++ridx) {
            for (int iy = 0; iy < img_height; ++iy) {

              for (int ix = 0; ix < strip_width; ++ix)
                log_part_detections[rootpart_idx][scaleidx][ridx][iy][ix] = LOG_ZERO;

              for (int ix = (img_width - strip_width); ix < img_width; ++ix) 
                log_part_detections[rootpart_idx][scaleidx][ridx][iy][ix] = LOG_ZERO;

            }// iy
              
          }// rotations

      }// strip border

      double scale = scale_from_index(part_app.m_exp_param, scaleidx);

      if (bDebugMessageOutput)
        cout << "processing scale " << scaleidx << "(" << scale << ")" << endl;

      /* after upstream pass: message from downstream combined with appearance model */
      /* after downstream pass: part posterior                                       */
      vector<FloatGrid3> log_part_posterior(nParts, 
                                            FloatGrid3(boost::extents[nRotations][img_height][img_width]));

      vector<bool> vComputedPosteriors(nParts, false);
      vector<int> compute_stack;
      compute_stack.push_back(rootpart_idx);

      /** these are only needed if we also want to compute part marginals */
      vector<FloatGrid3> log_from_root;
      if (bComputePartMarginals) 
        log_from_root.resize(nParts, FloatGrid3(boost::extents[nRotations][img_height][img_width]));


      /** these are only needed if we sample from posterior */
      vector<FloatGrid3> log_part_posterior_sample; 
      vector<FloatGrid3> log_from_root_sample;
      if (num_samples > 0) {
        log_part_posterior_sample.resize(nParts, FloatGrid3(boost::extents[nRotations][img_height][img_width]));
        log_from_root_sample.resize(nParts, FloatGrid3(boost::extents[nRotations][img_height][img_width]));
      }

      /**  pass messages from children to the root  */

      while (!compute_stack.empty()) {
        bool bCanCompute = true;

        int curidx = compute_stack.back();
        compute_stack.pop_back();
        
        if (bDebugMessageOutput) 
          cout << "curidx: " << curidx << endl;
        
        for (uint jidx = 0; jidx < joints.size(); ++jidx) {
          if (joints[jidx].parent_idx == curidx) {
            if (vComputedPosteriors[joints[jidx].child_idx] == false) {
              bCanCompute = false;
              compute_stack.push_back(curidx);
              compute_stack.push_back(joints[jidx].child_idx);

              if (bDebugMessageOutput)
                cout << "push: " << curidx << ", " << joints[jidx].child_idx << endl;
              break;
            }          
          }
        }// joints


        if (bCanCompute) {
          if (bDebugMessageOutput)
            cout << "computing posterior for " << curidx << endl;

          if (curidx == rootpart_idx) 
            assert(compute_stack.empty());

          vector<int> all_children;
          vector<int> incoming_joints;
          get_incoming_joints(joints, curidx, all_children, incoming_joints);
          
          for (int i = 0; i < (int)all_children.size(); ++i) {
            int child_idx = all_children[i];
            int jidx = incoming_joints[i];

            if (bDebugMessageOutput)
              cout << "\tcomputing component from " << child_idx << endl;
            
            assert(vComputedPosteriors[child_idx]);

            FloatGrid3 log_part_posterior_from_child(boost::extents[nRotations][img_height][img_width]);

            computeRotJointMarginal(part_app.m_exp_param, 
                                    log_part_posterior[child_idx], log_part_posterior_from_child, 
                                    joints[jidx].offset_c, joints[jidx].offset_p, 
                                    joints[jidx].C, 
                                    joints[jidx].rot_mean, joints[jidx].rot_sigma,
                                    scale, bIsSparse);

            multi_array_op::addGrid2(log_part_posterior[curidx], log_part_posterior_from_child);

            /* start computing messages to direct children of root node     */
            /* do not include message that was received from child          */
            if (curidx == rootpart_idx && bComputePartMarginals) {
              for (int i2 = 0; i2 < (int)all_children.size(); ++i2) {
                if (i2 != i) {
                  int child_idx2 = all_children[i2];
                  cout << "\tadding posterior from " << child_idx << " to root message to " << child_idx2 << endl;
                  multi_array_op::addGrid2(log_from_root[child_idx2], log_part_posterior_from_child);
                }
              }
            }


          }// children

          if (part_app.m_part_conf.part(curidx).is_detect()) {
            multi_array_op::addGrid2(log_part_posterior[curidx], log_part_detections[curidx][scaleidx]);
          }

          vComputedPosteriors[curidx] = true;
        }// if can compute
      }// stack

      /* store log_from_root before adding appearance of root part                                           */
      /* store precomputed products of appearance and messages from downstream in log_part_posterior_sample */
      if (num_samples > 0) {
        log_from_root_sample = log_from_root;
        log_part_posterior_sample = log_part_posterior;
      }

      /** pass messages from root to children */
      if (bComputePartMarginals) 
        computePartMarginals(part_app, joints, 
                            rootpart_idx, scaleidx, flip, imgidx,
                            log_part_detections, 
                            log_part_posterior, 
                            log_from_root);

      /** store root posterior (used for object detection) */
      root_part_posterior_full[scaleidx] = log_part_posterior[rootpart_idx];

      /** sample from posterior */

      if (num_samples > 0) {
        assert(false && "sampling from posterior not supported yet");
      }

    }// scales


    /** 
        since hypothesis with multiple orienations are not supported at the moment
        here we marginalize over valid orientations 

        in the end hypothesis is assumed to be upright 
     */

    vector<int> valid_root_rotations;

    if (part_app.m_part_conf.part(rootpart_idx).is_upright()) {
      int keep_idx1 = index_from_rot(part_app.m_exp_param, -1e-6);
      int keep_idx2 = index_from_rot(part_app.m_exp_param, 1e-6);
      cout << "keep only upright orientations of the root part: " << keep_idx1 << ", " << keep_idx2 << endl;

      valid_root_rotations.push_back(keep_idx1);
      if (keep_idx2 != keep_idx1)
        valid_root_rotations.push_back(keep_idx2);
    }
    else {
      cout << "keep all orientations of the root part" << endl;

      for (int rotidx = 0; rotidx < nRotations; ++rotidx) {
        valid_root_rotations.push_back(rotidx);
      }
      cout << endl;
    }

    assert(valid_root_rotations.size() > 0);

    root_part_posterior.resize(boost::extents[nScales][img_height][img_width]);
    root_part_posterior = root_part_posterior_full[boost::indices[index_range()][valid_root_rotations[0]][index_range()][index_range()]];
    multi_array_op::computeExpGrid(root_part_posterior);

    for (uint idx = 1; idx < valid_root_rotations.size(); ++idx) {
      FloatGrid3 tmp_grid = root_part_posterior_full[boost::indices[index_range()][valid_root_rotations[idx]][index_range()][index_range()]];

      multi_array_op::computeExpGrid(tmp_grid);
      multi_array_op::addGrid2(root_part_posterior, tmp_grid);
    }

    multi_array_op::computeLogGrid(root_part_posterior);
  }

  void findObjectImageRotJoints(const PartApp &part_app, int imgidx, bool flip, HypothesisList &hypothesis_list, int scoreProbMapType)
  {
    cout << "findObjectImageRotJoints" << endl;

    const PartConfig &part_conf = part_app.m_part_conf;
    bool bIsSparse = true; 

    /** load joints */
    int nJoints = part_conf.joint_size();
    int nParts = part_conf.part_size();
    int nScales = part_app.m_exp_param.num_scale_steps();
    int nRotations = part_app.m_exp_param.num_rotation_steps();

    cout << "nParts: " << nParts << endl;
    cout << "nJoints: " << nJoints << endl;

    // yes load the image again :) just to find its width
    int img_width, img_height;
    {
      kma::ImageContent *kmaimg = kma::load_convert_gray_image(part_app.m_test_annolist[imgidx].imageName().c_str());
      assert(kmaimg != 0);
      img_width = kmaimg->x();
      img_height = kmaimg->y();
      delete kmaimg;
    }

    vector<Joint> joints(nJoints);
    loadJoints(part_app, joints, flip);

    for (int jidx = 0; jidx < nJoints; ++jidx) {
      assert(joints[jidx].type == Joint::ROT_GAUSSIAN);

      cout << "covariance matrix of the joint: " << endl;
      boost_math::print_matrix(joints[jidx].C);
      // test end

    }

    /** load classifier scores */
    cout << "nParts: " << nParts << endl;
    cout << "nScales: " << nScales << endl;
    cout << "nRotations: " << nRotations << endl;
    cout << "img_height: " << img_height << endl;
    cout << "img_width: " << img_width << endl;

    double mb_count = (4.0 * nParts * nScales * nRotations * img_height * img_width) / (square(1024));
    cout << "allocating: " << mb_count << " MB " << endl;
    vector<vector<FloatGrid3> > log_part_detections(nParts, 
                                                    vector<FloatGrid3>(nScales, 
                                                                       FloatGrid3(boost::extents[nRotations][img_height][img_width])));
    cout << "done." << endl;
                                                                                                
    assert(scoreProbMapType == SPMT_NONE);

    cout << "loading scores ... " << endl;

    int rootpart_idx = -1;
    for (int pidx = 0; pidx < nParts; ++pidx) {
      if (part_conf.part(pidx).is_detect()) {

        bool bInterpolate = false;

        vector<vector<FloatGrid2> > cur_part_detections;

        part_app.loadScoreGrid(cur_part_detections, imgidx, pidx, flip, bInterpolate);

        assert((int)cur_part_detections.size() == nScales);
        assert((int)cur_part_detections[0].size() == nRotations);

        for (int scaleidx = 0; scaleidx < nScales; ++scaleidx) 
          for (int rotidx = 0; rotidx < nRotations; ++rotidx) {
            log_part_detections[pidx][scaleidx][rotidx] = cur_part_detections[scaleidx][rotidx];
          }

      }// if is_detect

      if (part_conf.part(pidx).is_root()) {
        assert(rootpart_idx == -1);
        rootpart_idx = pidx;
      }
    }

    cout << "done." << endl;

    assert(rootpart_idx >= 0 && "root part not found");
    cout << endl << "rootpart_idx: " << rootpart_idx << endl << endl;;

    for (int pidx = 0; pidx < nParts; ++pidx) 
      if (part_app.m_part_conf.part(pidx).is_detect()) {
        for (int scaleidx = 0; scaleidx < nScales; ++scaleidx) {

          if (scoreProbMapType == SPMT_NONE)
            object_detect::clip_scores_fill(log_part_detections[pidx][scaleidx]);
          else 
            assert(false);

          multi_array_op::computeLogGrid(log_part_detections[pidx][scaleidx]);
        }
      }
      
    FloatGrid3 root_part_posterior;

    computeRootPosteriorRot(part_app, log_part_detections, root_part_posterior, rootpart_idx, joints, flip, 
                            bIsSparse, imgidx);

    int max_hypothesis_number = 1000;
    findLocalMax(part_app.m_exp_param, root_part_posterior, hypothesis_list, max_hypothesis_number);

    for (int hypidx = 0; hypidx < hypothesis_list.hyp_size(); ++hypidx) {
      hypothesis_list.mutable_hyp(hypidx)->set_flip(flip);

      /** these are hypothesis for root part, convert them to hypothesis for object bounding box */
      int bbox_x = (int)(hypothesis_list.hyp(hypidx).x() + part_app.m_window_param.bbox_offset_x());
      int bbox_y = (int)(hypothesis_list.hyp(hypidx).y() + part_app.m_window_param.bbox_offset_y());
      hypothesis_list.mutable_hyp(hypidx)->set_x(bbox_x);
      hypothesis_list.mutable_hyp(hypidx)->set_y(bbox_y);
    }

    for (int i = 0; i < min(10, hypothesis_list.hyp_size()); ++i) {
      cout << "hypothesis " << i << 
        ", x: " << hypothesis_list.hyp(i).x() << 
        ", y: " << hypothesis_list.hyp(i).y() << 
        ", scaleidx: " << index_from_scale(part_app.m_exp_param, hypothesis_list.hyp(i).scale()) << 
        ", score: " << hypothesis_list.hyp(i).score() << endl;
    }
  }



}// namespace
