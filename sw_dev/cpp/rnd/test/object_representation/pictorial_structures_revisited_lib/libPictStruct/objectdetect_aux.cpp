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

// boost::random (needed for sampling from discrete distributions)
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/uniform_real.hpp>

#include <boost/lambda/lambda.hpp>
#include <boost/lambda/bind.hpp>

#include <libProtoBuf/protobuf_aux.hpp>

#include <libPartApp/partapp_aux.hpp>

#include <libMisc/misc.hpp>

#include "HypothesisList.pb.h"

#include "objectdetect.h"

using namespace std;

using boost_math::double_vector;
using boost_math::double_matrix;
using boost_math::double_zero_matrix;

using boost::multi_array_types::index_range;

using namespace boost::lambda;
using namespace std;

namespace object_detect 
{

  void loadJoints(const PartApp &part_app, vector<Joint> &joints, bool flip)
  {
    const PartConfig &part_conf = part_app.m_part_conf;

    int nJoints = part_conf.joint_size();
    cout << "nJoints: " << nJoints << endl;
    joints.clear();
    joints.resize(nJoints);

    int nParts = part_conf.part_size();
    cout << "nParts: " << nParts << endl;

    /** test that part_id = part_index + 1*/
    for (int pidx = 0; pidx < nParts; ++pidx) {
      assert(part_conf.part(pidx).part_id() == pidx + 1);
    }

    /** load joint parameters, convert joints to the correct flip if needed    */
    for (int jidx = 0; jidx < nJoints; ++jidx) {
      load_joint(part_app, jidx, joints[jidx]);
      assert(joints[jidx].C.size1() == 2 && joints[jidx].C.size2() == 2);

      if (flip) {
        double_matrix T = boost_math::double_identity_matrix(2);
        T(0, 0) = -1;

        assert(joints[jidx].type == Joint::POS_GAUSSIAN || joints[jidx].type == Joint::ROT_GAUSSIAN);

        double_matrix C = prod(T, double_matrix(prod(joints[jidx].C, T)));
        double_vector offset_p = prod(T, joints[jidx].offset_p);
        double_vector offset_c = prod(T, joints[jidx].offset_c);
        joints[jidx].C = C;
        joints[jidx].offset_p = offset_p;
        joints[jidx].offset_c = offset_c;

        if (joints[jidx].type == Joint::ROT_GAUSSIAN) {
          joints[jidx].rot_mean = -joints[jidx].rot_mean;
        }
      }

      cout << "offset_p: " << jidx << endl;
      boost_math::print_vector(joints[jidx].offset_p);
      
      /** parent and child parts are in terms of "part id", convert it to index in the "part_app.m_part_conf.part" */
      joints[jidx].parent_idx--;
      joints[jidx].child_idx--;

      assert(joints[jidx].child_idx >= 0 && joints[jidx].child_idx < nParts);
      assert(joints[jidx].parent_idx >= 0 && joints[jidx].parent_idx < nParts);
    }

    assert((int)joints.size() == nParts - 1);    
  }


  int JointTypeFromString(QString qsType)
  {
    if (qsType == "Gaussian" || qsType == "PosGaussian") {
      return Joint::POS_GAUSSIAN;
    }
    else if (qsType == "RotGaussian") {
      return Joint::ROT_GAUSSIAN;
    }
    else {
      assert(false && "unknown joint type");
    }
    return -1;
  }


  int MapTypeFromString(QString qsScoreProbMapType) 
  {
    if (qsScoreProbMapType == "none") 
      return SPMT_NONE;
    else if (qsScoreProbMapType == "ratio") 
      return SPMT_RATIO;
    else if (qsScoreProbMapType == "ratio_prior") 
      return SPMT_RATIO_WITH_PRIOR;
    else if (qsScoreProbMapType == "spatial") 
      return SPMT_SPATIAL;
    else if (qsScoreProbMapType == "spatial_prod")
      return SPMT_SPATIAL_PROD;
    else 
      assert(false && "unknown map type");

    return -1;
  }

  QString MapTypeToString(int map_type) 
  {
    if (map_type == SPMT_NONE) 
      return "none";
    else if (map_type == SPMT_RATIO) 
      return "ratio";
    else if (map_type == SPMT_RATIO_WITH_PRIOR) 
      return "ratio_prior";
    else if (map_type == SPMT_SPATIAL) 
      return "spatial";
    else if (map_type == SPMT_SPATIAL_PROD)
      return "spatial_prod";
    else 
      assert(false && "unknown map type");

    return "";
  }

//--S [] 2012/07/24: Sang-Wook Lee
  bool ScorePredicate(const ObjectHypothesis &lhs, const ObjectHypothesis &rhs)
  {
	  return lhs.score() > rhs.score();    
  }
//--E [] 2012/07/24

  void findLocalMax(const ExpParam &exp_param, const FloatGrid3 &log_prob_grid, 
                    HypothesisList &hypothesis_list,
                    int max_hypothesis_number)
  {
    cout << "findLocalMax, max_hypothesis_number: " << max_hypothesis_number << endl;

    int nImageWidth = log_prob_grid.shape()[2];
    int nImageHeight = log_prob_grid.shape()[1];
    int nScales = log_prob_grid.shape()[0];

    bool bScaleMax = true;

    vector<ObjectHypothesis> vLocalMax;

    for (int sidx = 0; sidx < nScales; ++sidx) {
      //     float minval, maxval;
      //     multi_array_op::getMinMax(log_prob_grid[sidx], minval, maxval);
    
      //     cout << "scale: " << sidx 
      //          << " min: " << minval
      //          << " max: " << maxval 
      //          << endl;
      
      for (int x = 0; x < nImageWidth; ++x)
        for (int y = 0; y < nImageHeight; ++y) {
          bool bLocalMaxima = true;
        
          for (int dy = -1; dy <= 1 && bLocalMaxima; ++dy) { 
            for (int dx = -1; dx <= 1 && bLocalMaxima; ++dx) {
              int xpos = x + dx;
              int ypos = y + dy;
              if (xpos >= 0 && xpos < nImageWidth && ypos >= 0 && ypos < nImageHeight && !(dx == 0 && dy == 0)) {
                if (log_prob_grid[sidx][ypos][xpos] > log_prob_grid[sidx][y][x]) 
                  bLocalMaxima = false;
              }
            }
          }
          
          if (bScaleMax) {
            if (bLocalMaxima && sidx > 0)
              bLocalMaxima = log_prob_grid[sidx-1][y][x] < log_prob_grid[sidx][y][x];

            if (bLocalMaxima && sidx < nScales - 1)
              bLocalMaxima = log_prob_grid[sidx+1][y][x] < log_prob_grid[sidx][y][x];
          }

          if (bLocalMaxima) {
            ObjectHypothesis h;
            h.set_x(x);
            h.set_y(y);
            h.set_scale(scale_from_index(exp_param, sidx));
            h.set_score(log_prob_grid[sidx][y][x]);
            vLocalMax.push_back(h);
          }

        }// position
    }// scale

	//--S [] 2012/07/24: Sang-Wook Lee
    //std::sort(vLocalMax.begin(), vLocalMax.end(), 
    //          bind(&ObjectHypothesis::score, _1) > bind(&ObjectHypothesis::score, _2));    
    std::sort(vLocalMax.begin(), vLocalMax.end(), ScorePredicate);    
	//--E [] 2012/07/24

    if (vLocalMax.size() > (uint)max_hypothesis_number)
      vLocalMax.erase(vLocalMax.begin() + max_hypothesis_number, vLocalMax.end());
  
    cout << "found " << vLocalMax.size() << " local maxima. " << endl;

    for (int hidx = 0; hidx < (int)vLocalMax.size(); ++hidx) {
      ObjectHypothesis *h = hypothesis_list.add_hyp();
      (*h) = vLocalMax[hidx];
    }
  }

  QString getObjectHypFilename(int imgidx, bool flip, int scoreProbMapType) {
    QString qsFilename = "/object_hyp_imgidx" + 
      QString::number(imgidx) + 
      "_o" + QString::number((int)flip) +
      "_spm" + MapTypeToString(scoreProbMapType);

    qsFilename += ".pbuf";

    return qsFilename;
  }

  void findObjectDataset(const PartApp &part_app, int firstidx, int lastidx, int scoreProbMapType)
  {
    cout << "findObjectDataset" << endl;
    //bool bLoadLogP = false;
    //int lp_version = 0;

    assert(firstidx >= 0 && firstidx <= (int)part_app.m_test_annolist.size());
    assert(lastidx < (int)part_app.m_test_annolist.size());

    QString qsHypDir = (part_app.m_exp_param.log_dir() + "/" + 
                        part_app.m_exp_param.log_subdir() + "/object_hyp").c_str();


    if (!filesys::check_dir(qsHypDir)) {
      cout << "creating " << qsHypDir.toStdString() << endl;
      assert(filesys::create_dir(qsHypDir));
    }

    /** find out what type of joints are used in the spatial model (currenly no models with heterogeneous joints are supported) */
    bool bFindObjectRot = false;
    if (part_app.m_part_conf.joint_size() > 0) {
      Joint joint;
      int jidx = 0;
      load_joint(part_app, jidx, joint);
      if (joint.type == Joint::ROT_GAUSSIAN)
        bFindObjectRot = true;
    }

    cout << "processing images " << firstidx << " to " << lastidx << endl;
    cout << "bFindObjectRot: " << bFindObjectRot << endl;

    for (int imgidx = firstidx; imgidx <= lastidx; ++imgidx) {
      cout << "processing image " << imgidx << endl;

      int flip_count = 1;
      if (part_app.m_exp_param.flip_orientation())
        ++flip_count;

      /** test for orientation as in trining set (0) and reverse orientation (1)*/
      for (int flip = 0; flip < flip_count; ++flip) {
        HypothesisList hypothesis_list;

        if (bFindObjectRot) {
          findObjectImageRotJoints(part_app, imgidx, flip, hypothesis_list, scoreProbMapType); 
        }
        else {
          findObjectImagePosJoints(part_app, imgidx, flip, hypothesis_list, scoreProbMapType);
        }

        QString qsFilename = qsHypDir + getObjectHypFilename(imgidx, flip, scoreProbMapType);

        cout << "saving " << qsFilename.toStdString() << endl;
        write_message_binary(qsFilename, hypothesis_list);
      }

    }// images
  }

  void nms_recursive(const HypothesisList hypothesis_list, 
                     vector<bool> &nms, 
                     double train_object_width, 
                     double train_object_height) 
  {
    double dist_threshold = 1.0;

    /** used for comparison with tracklets detector, used in supplementary material to CVPR'09 paper */
    //double ellipse_size = 0.3;
    
    /** used in all other experiments */
    double ellipse_size = 0.5;

    int nHypothesis = hypothesis_list.hyp_size();

    cout << "nms_recursive" << endl;
    cout << "number of hypothesis: " << nHypothesis << endl;
  
    nms.clear();
    nms.resize(nHypothesis, false);

    for (int i = 0; i < nHypothesis; ++i) {
      if (i != nHypothesis - 1) {
        if (hypothesis_list.hyp(i).score() < hypothesis_list.hyp(i+1).score()) {
          cout << i << " " << hypothesis_list.hyp(i).score() << " " << hypothesis_list.hyp(i+1).score() << endl;
          assert(false && "unsorted hypothesis vector");
        }
      }

      double scale = hypothesis_list.hyp(i).scale();

      for (int j = i+1; j < nHypothesis; ++j) {
        double e1 = square(ellipse_size*scale*train_object_width);
        double e2 = square(ellipse_size*scale*train_object_height);
        
        double dx = hypothesis_list.hyp(i).x() - hypothesis_list.hyp(j).x();
        double dy = hypothesis_list.hyp(i).y() - hypothesis_list.hyp(j).y();

        if (dx*dx/e1 + dy*dy/e2 < dist_threshold) {
          nms[j] = true;
        }
      }
    }
    cout << "done." << endl;
  }
  
  
  void saveRecoResults(const PartApp &part_app, int scoreProbMapType)
  {
    AnnotationList resultsAnnoListAll;
    AnnotationList resultsAnnoListNms;
    vector<int> vEmptyImages;

    QString qsHypDir = (part_app.m_exp_param.log_dir() + "/" + 
                        part_app.m_exp_param.log_subdir() + "/object_hyp").c_str();

    assert(filesys::check_dir(qsHypDir));

    int nImages = part_app.m_test_annolist.size();
    for (int imgidx = 0; imgidx < nImages; ++imgidx) {
      cout << "imgidx: " << imgidx << endl;

      vector<ObjectHypothesis> v_hypothesis;
      bool bComplete = false;

      bool flip = false;
      QString qsFilename0 = qsHypDir + getObjectHypFilename(imgidx, flip, scoreProbMapType);

      if (filesys::check_file(qsFilename0)) {
        /** load hypothesis */
        HypothesisList hypothesis_list0;
        assert(parse_message_binary(qsFilename0, hypothesis_list0));

        for (int hypidx = 0; hypidx < hypothesis_list0.hyp_size(); ++hypidx)
          v_hypothesis.push_back(hypothesis_list0.hyp(hypidx));

        if (part_app.m_exp_param.flip_orientation()) {
          flip = true;
          QString qsFilename1 = qsHypDir + getObjectHypFilename(imgidx, flip, scoreProbMapType);

          if (filesys::check_file(qsFilename1)) {
            HypothesisList hypothesis_list1;
            assert(parse_message_binary(qsFilename1, hypothesis_list1));

            for (int hypidx = 0; hypidx < hypothesis_list1.hyp_size(); ++hypidx)
              v_hypothesis.push_back(hypothesis_list1.hyp(hypidx));

            bComplete = true;
          }
        }
        else {
          bComplete = true;
        }
      }

      if (bComplete) {
        /** merge and sort the hypothesis from different orientations */
		//--S [] 2012/07/24: Sang-Wook Lee
        //std::sort(v_hypothesis.begin(), v_hypothesis.end(), 
        //          bind(&ObjectHypothesis::score, _1) > bind(&ObjectHypothesis::score, _2));   
		std::sort(v_hypothesis.begin(), v_hypothesis.end(), ScorePredicate);    
		//--E [] 2012/07/24
     
        HypothesisList hypothesis_list;
        for (int hypidx = 0; hypidx < (int)v_hypothesis.size(); ++hypidx) {
          ObjectHypothesis *h = hypothesis_list.add_hyp();
          (*h) = v_hypothesis[hypidx];
        }

        /** save hypothesis as idl and al */
        Annotation resultsAnnoAll(part_app.m_test_annolist[imgidx].imageName());
        Annotation resultsAnnoNms(part_app.m_test_annolist[imgidx].imageName());

        /** do non-maximum suppression */
        vector<bool> nms;
        double train_height = part_app.m_window_param.train_object_height();
        double train_width = train_height / part_app.m_exp_param.object_height_width_ratio();
        nms_recursive(hypothesis_list, nms, 
                      train_width,
                      train_height);
               

        for (int hypidx = 0; hypidx < hypothesis_list.hyp_size(); ++hypidx) {
          int nRectHeight = (int)(hypothesis_list.hyp(hypidx).scale() * part_app.m_window_param.train_object_height());
          //int nRectWidth = (int)(hypothesis_list.hyp(hypidx).scale() * part_app.m_window_param.train_object_width());
          int nRectWidth = (int)(hypothesis_list.hyp(hypidx).scale() * part_app.m_window_param.train_object_height() / 
                                 part_app.m_exp_param.object_height_width_ratio());

          AnnoRect r((int)(hypothesis_list.hyp(hypidx).x() - nRectWidth/2), (int)(hypothesis_list.hyp(hypidx).y() - nRectHeight/2), 
                     (int)(hypothesis_list.hyp(hypidx).x() + nRectWidth/2), (int)(hypothesis_list.hyp(hypidx).y() + nRectHeight/2),
                     hypothesis_list.hyp(hypidx).score(),
                     (int)hypothesis_list.hyp(hypidx).flip(), 
                     hypothesis_list.hyp(hypidx).scale());

          resultsAnnoAll.addAnnoRect(r);
          /** begin temp edit (-200) */
          if (!nms[hypidx]) {
            if (hypothesis_list.hyp(hypidx).score() > -200)
              resultsAnnoNms.addAnnoRect(r);
            else 
              cout << "warning: ignoring hypothesis with score < -200" << endl;
          }
        
          /** end temp edit */

        }// hypothesis

        resultsAnnoListNms.addAnnotation(resultsAnnoNms);
        resultsAnnoListAll.addAnnotation(resultsAnnoAll);
      }// if complete
      else {
        vEmptyImages.push_back(imgidx);
      }


    }// images


    /** show images which are still not processed */
    for_each(vEmptyImages.begin(), vEmptyImages.end(), cout << _1 << " ");
    cout << endl << "#images without detections: " << vEmptyImages.size() << endl;

    QString qsResultDir = (part_app.m_exp_param.log_dir() + "/" + 
                           part_app.m_exp_param.log_subdir() + "/results").c_str();
    if (!filesys::check_dir(qsResultDir)) {
      cout << "creating " << qsResultDir.toStdString() << endl;
      assert(filesys::create_dir(qsResultDir));
    }

    QString qsFilenameBase = qsResultDir + "/" + part_app.m_exp_param.log_subdir().c_str() + 
      "_spm" + MapTypeToString(scoreProbMapType);


    /** save if at list for some images results are available */
    if (vEmptyImages.size() != part_app.m_test_annolist.size()) {
      bool bSaveRelativeToHome = true;
      
      //resultsAnnoListAll.save(qsFilenameBase.toStdString() + "-all.al");
      //cout << endl;
      resultsAnnoListAll.save(qsFilenameBase.toStdString() + "-all.idl", bSaveRelativeToHome);
      cout << endl;

      resultsAnnoListNms.save(qsFilenameBase.toStdString() + "-nms.al");
      cout << endl;
      resultsAnnoListNms.save(qsFilenameBase.toStdString() + "-nms.idl", bSaveRelativeToHome);
      cout << endl;
    }
    
    cout << "test_dataset: " << part_app.m_exp_param.test_dataset(0) << endl;
  }

}// namespace
