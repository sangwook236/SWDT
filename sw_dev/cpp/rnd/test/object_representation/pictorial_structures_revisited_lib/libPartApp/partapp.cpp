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

#include <iostream>

#include <QDir>

#include <libAnnotation/annotationlist.h>

#include <libFilesystemAux/filesystem_aux.h>
#include <libPartDetect/partdetect.h>

#include <libProtoBuf/protobuf_aux.hpp>

#include <libMatlabIO/matlab_io.h>
#include <libMatlabIO/matlab_io.hpp>
#include <libMatlabIO/matlab_cell_io.hpp>

#include <libMultiArray/multi_array_op.hpp>
#include <libMultiArray/multi_array_transform.hpp>

#include "partapp.h"
#include "partapp_aux.hpp"

using namespace std;

using boost::multi_array_types::index_range;

void bbox_from_pos(const ExpParam &exp_param, const PartWindowParam::PartParam &part_param, 
                   int scaleidx, int rotidx, int ix, int iy, 
                   PartBBox &bbox)
{
  bbox.part_pos(0) = ix;
  bbox.part_pos(1) = iy;

  double rot = rot_from_index(exp_param, rotidx) / 180.0 * M_PI;
  bbox.part_x_axis(0) = cos(rot);
  bbox.part_x_axis(1) = sin(rot);

  bbox.part_y_axis(0) = -bbox.part_x_axis(1);
  bbox.part_y_axis(1) = bbox.part_x_axis(0);

  double scale = scale_from_index(exp_param, scaleidx);
  double rect_width = scale*part_param.window_size_x();
  double rect_height = scale*part_param.window_size_y();
  //cout << "bbox_from_pos, scale: " << scale << ", rect_width: " << rect_width << ", rect_height: " << rect_height << endl;

  bbox.min_proj_x = -scale*part_param.pos_offset_x();
  bbox.min_proj_y = -scale*part_param.pos_offset_y();

  bbox.max_proj_x = bbox.min_proj_x + rect_width;
  bbox.max_proj_y = bbox.min_proj_y + rect_height;
}

void appendAnnoList(AnnotationList &annolist1, const AnnotationList &annolist2)
{
  for (int aidx = 0; aidx < (int)annolist2.size(); ++aidx) 
    annolist1.addAnnotation(annolist2[aidx]);
}

void convertFullPath(QString qsAnnoListFile, AnnotationList &annolist) {
  assert(filesys::check_file(qsAnnoListFile) && "annotation file not found");

  QString qsAnnoListPath;
  QString qsAnnoListName;
  filesys::split_filename(qsAnnoListFile, qsAnnoListPath, qsAnnoListName);

  for (int imgidx = 0; imgidx < (int)annolist.size(); ++imgidx) {
    if (!filesys::check_file(annolist[imgidx].imageName().c_str())) {

      QString qsFilename = qsAnnoListPath + "/" + annolist[imgidx].imageName().c_str();

      if (!filesys::check_file(qsFilename)) {
        cout << "file not found: " << qsFilename.toStdString() << endl;
        assert(false);
      }

      annolist[imgidx].setImageName(qsFilename.toStdString());
    }

  }
}

QString complete_relative_path(QString qsInputFile, QString qsReferenceFile)
{
  qsInputFile = qsInputFile.trimmed();

  QDir dir(qsInputFile);

  QString qsRes;

  if (dir.isRelative()) {
    QString qsPath1, qsName1;
    QString qsPath2, qsName2;

    filesys::split_filename(qsInputFile, qsPath1, qsName1);
    filesys::split_filename(qsReferenceFile, qsPath2, qsName2);

//     if (qsPath1 == ".")
//       qsRes = qsPath2 + "/" + qsName1;
//     else 
//       qsRes = qsPath2 + qsPath1.mid(1) + qsName1;

    /** corrections from Marcin */
    if (qsPath1 == ".")
      qsRes = qsPath2 + "/" + qsName1;
    else 
      qsRes = qsPath2 + qsPath1.mid(1) + "/" + qsName1;

  }
  else {
    qsRes = qsInputFile;
  }

  return qsRes;
}

void PartApp::init(QString qsExpParam)
{
  parse_message_from_text_file(qsExpParam, m_exp_param);

  assert(m_exp_param.part_conf().length() > 0);
  assert(m_exp_param.abc_param().length() > 0);
  assert(m_exp_param.log_dir().length() > 0);
  assert(m_exp_param.train_dataset_size() > 0);

  /* load part configuration and detector parameters */
  QString qsPartConf = complete_relative_path(m_exp_param.part_conf().c_str(), qsExpParam);
  QString qsAbcParam = complete_relative_path(m_exp_param.abc_param().c_str(), qsExpParam);

  cout << "part configuration: " << qsPartConf.toStdString() << endl;
  cout << "classifier parameters: " << qsAbcParam.toStdString() << endl;

  parse_message_from_text_file(qsPartConf, m_part_conf);
  parse_message_from_text_file(qsAbcParam, m_abc_param);
  
  assert(m_part_conf.part_size() > 0 && "missing part definitions");

  if (!m_exp_param.has_log_subdir()) {
    QString qsExpParamPath;
    QString qsExpParamName;
    QString qsExpParamExt;
    filesys::split_filename_ext(qsExpParam, qsExpParamPath, qsExpParamName, qsExpParamExt);
    m_exp_param.set_log_subdir(qsExpParamName.toStdString());
  }

  if (!m_exp_param.has_class_dir()) {
    m_bExternalClassDir = false;

    m_exp_param.set_class_dir(m_exp_param.log_dir() + "/" + m_exp_param.log_subdir() + "/class");
    cout << "set class_dir to: " << m_exp_param.class_dir() << endl;
  }
  else {
    m_bExternalClassDir = true;

    cout << "class_dir: " << m_exp_param.class_dir() << endl;
  }

  /** initialize scoregrid_dir to default value */
  if (!m_exp_param.has_scoregrid_dir()) {
    m_exp_param.set_scoregrid_dir(m_exp_param.log_dir() + "/" + m_exp_param.log_subdir() + "/test_scoregrid");
    cout << "set scoregrid_dir to: " << m_exp_param.scoregrid_dir() << endl;
  }
  else {
    cout << "scoregrid_dir: " << m_exp_param.scoregrid_dir() << endl;
  }

  /** initialize partprob_dir to default value */
  if (!m_exp_param.has_partprob_dir()) {
    m_exp_param.set_partprob_dir(m_exp_param.log_dir() + "/" + m_exp_param.log_subdir() + "/test_partprob");
    cout << "set partprob_dir to: " << m_exp_param.partprob_dir() << endl;
  }
  else {
    cout << "partprob_dir: " << m_exp_param.partprob_dir() << endl;
  }

  if (!filesys::check_dir(m_exp_param.log_dir().c_str())) {
    assert(filesys::create_dir(m_exp_param.log_dir().c_str()));
  }

  if (!filesys::check_dir(m_exp_param.class_dir().c_str())) {
    assert(filesys::create_dir(m_exp_param.class_dir().c_str()));
  }

  /* load training, validation and test data */
  if (m_exp_param.train_dataset_size() > 0) {

    /* concatenate training annotation files */
    for (int listidx = 0; listidx < m_exp_param.train_dataset_size(); ++listidx) {
      assert(filesys::check_file(m_exp_param.train_dataset(listidx).c_str()));

      cout << "\nloading training data from " << m_exp_param.train_dataset(listidx) << endl;

      AnnotationList annolist;
      annolist.load(m_exp_param.train_dataset(listidx));

      appendAnnoList(m_train_annolist, annolist);
    }

    //assert(m_train_annolist.size() > 0);
  }
  cout << "loaded " << m_train_annolist.size() << " training images" << endl;

  if (m_exp_param.validation_dataset_size() > 0) {

    for (int listidx = 0; listidx < m_exp_param.validation_dataset_size(); ++listidx) {
      assert(filesys::check_file(m_exp_param.validation_dataset(listidx).c_str()));

      cout << "\nloading validataion data from " << m_exp_param.validation_dataset(listidx) << endl;

      AnnotationList annolist;
      annolist.load(m_exp_param.validation_dataset(listidx));

      appendAnnoList(m_validation_annolist, annolist);
    }

    //assert(m_validation_annolist.size() > 0);
  }

  if (m_exp_param.test_dataset_size() > 0) {

    /* concatenate test annotation files */
    for (int listidx = 0; listidx < m_exp_param.test_dataset_size(); ++listidx) {
      assert(filesys::check_file(m_exp_param.test_dataset(listidx).c_str()));      

      cout << "\nloading test data from " << m_exp_param.test_dataset(listidx) << endl;      

      AnnotationList annolist;
      annolist.load(m_exp_param.test_dataset(listidx));

      /* expand path, in case it is given relative to home directory */
      convertFullPath(m_exp_param.test_dataset(listidx).c_str(), annolist);

      appendAnnoList(m_test_annolist, annolist);
    }

    assert(m_test_annolist.size() > 0);
  }

  QString qsWindowParamFile = (m_exp_param.class_dir() + "/window_param.txt").c_str();
  cout << "qsWindowParamFile: " << qsWindowParamFile.toStdString() << endl;

  /* load part window dimensions */
  
  if (filesys::check_file(qsWindowParamFile)) {
    cout << "\nloading window parameters from " << qsWindowParamFile.toStdString() << endl;
    parse_message_from_text_file(qsWindowParamFile, m_window_param);
  }
  else {
    cout << "WARNING: window parameters file not found" << endl;
  }
  
  /* compute part window dimensions if needed */
  if (m_window_param.part_size() != m_part_conf.part_size()) {
    cout << "\nrecomputing part window size ..." << endl;
  
    part_detect::compute_part_window_param(m_train_annolist, m_part_conf, m_window_param);
    assert(m_window_param.part_size() == m_part_conf.part_size());
    
    cout << "saving " << qsWindowParamFile.toStdString() << endl;
    print_message_to_text_file(qsWindowParamFile, m_window_param);
  }

  /**
     find the root part
   */
  for (int pidx = 0; pidx < m_part_conf.part_size(); ++pidx) {
    if (m_part_conf.part(pidx).is_root()) {
      assert(m_rootpart_idx == -1);
      m_rootpart_idx = pidx;
      break;
    }
  }
  assert(m_rootpart_idx >=0 && "missing root part");

  /**  
    save part parameters to enable visualization in Matlab
  */

  //if (!m_bExternalClassDir) {
  {
    QString qsParamsMatDir = (m_exp_param.log_dir() + "/" + m_exp_param.log_subdir() + "/params_mat").c_str();
    
    if (!filesys::check_dir(qsParamsMatDir))
      filesys::create_dir(qsParamsMatDir);

    cout << "saving Matlab visualization parameters in:  " << qsParamsMatDir.toStdString() << endl;

    /* here we skip parts which are not detected */

    int nParts = 0;
    for (int pidx = 0; pidx < m_part_conf.part_size(); ++pidx)
      if (m_part_conf.part(pidx).is_detect())
        ++nParts;

    matlab_io::mat_save_double(qsParamsMatDir + "/num_parts.mat", "num_parts", nParts);

    boost_math::double_matrix part_dims(nParts, 2);
    for (int pidx = 0; pidx < m_part_conf.part_size(); ++pidx)
      if (m_part_conf.part(pidx).is_detect()) {
        assert(m_window_param.part_size() > pidx);
        part_dims(pidx, 0) = m_window_param.part(pidx).window_size_x();
        part_dims(pidx, 1) = m_window_param.part(pidx).window_size_y();
      }

    matlab_io::mat_save_double_matrix(qsParamsMatDir + "/part_dims.mat", "part_dims", part_dims);

    MATFile *f = matlab_io::mat_open(qsParamsMatDir + "/rotation_params.mat", "wz");
    assert(f != 0);
    matlab_io::mat_save_double(f, "min_part_rotation", m_exp_param.min_part_rotation());
    matlab_io::mat_save_double(f, "max_part_rotation", m_exp_param.max_part_rotation());
    matlab_io::mat_save_double(f, "num_rotation_steps", m_exp_param.num_rotation_steps());
    matlab_io::mat_close(f);

    /**
       not sure what happens if root part is a "dummy" part without detector, 

       above we saved parameters only for detectable parts

     */

    matlab_io::mat_save_double(qsParamsMatDir + "/rootpart_idx.mat", "rootpart_idx", m_rootpart_idx);
  }
  

}

QString PartApp::getClassFilename(int pidx, int bootstrap_type) const
{
  QString qsClassFilename;

  if (bootstrap_type == 0) {
    qsClassFilename = m_exp_param.class_dir().c_str() + QString("/part") + QString::number(pidx) + ".class";
  }
  else if (bootstrap_type == 1) {
    qsClassFilename = m_exp_param.class_dir().c_str() + QString("/part") + QString::number(pidx) + "-bootstrap.class";
  }
  else {
    QString qsClassFilenameBootstrap = m_exp_param.class_dir().c_str() + QString("/part") + QString::number(pidx) + "-bootstrap.class";
    QString qsClassFilenameNormal = m_exp_param.class_dir().c_str() + QString("/part") + QString::number(pidx) + ".class";

    //if (m_abc_param.num_train_bootstrap() > 0) {

    //if (m_abc_param.bootstrap_fraction() > 0) {

    /** 
        "num_train_bootstrap" is deprecated, 
        at some point we should look at "bootstrap_fraction" only

        for now check both "bootstrap_fraction" and "num_train_bootstrap"
        in order to be able to load old bootstrapped classfiers
    */


    if (m_abc_param.bootstrap_fraction() > 0 || m_abc_param.num_train_bootstrap() > 0) {
      if (filesys::check_file(qsClassFilenameBootstrap)) {
        qsClassFilename = qsClassFilenameBootstrap;
      }
      else {
        cout << "warning: bootstrap classifier not found!!!" << endl;
        qsClassFilename = qsClassFilenameNormal;
      }
    }
    else {
      qsClassFilename = qsClassFilenameNormal;
    }
  }
  
  return qsClassFilename;
}

void PartApp::loadClassifier(AdaBoostClassifier &abc, int pidx, int bootstrap_type) const
{
  assert(m_exp_param.class_dir().length() > 0);
  assert(filesys::check_dir(m_exp_param.class_dir().c_str()));

  QString qsClassFilename = getClassFilename(pidx, bootstrap_type);

  cout << "loading classifier from " << qsClassFilename.toStdString() << endl;
  assert(filesys::check_file(qsClassFilename));

  abc.loadClassifier(qsClassFilename.toStdString());
}

void PartApp::saveClassifier(AdaBoostClassifier &abc, int pidx, int bootstrap_type) const
{
  assert(!m_bExternalClassDir && "can not update external parameters");
  
  assert(m_exp_param.class_dir().length() > 0);
  assert(filesys::check_dir(m_exp_param.class_dir().c_str()));

  //QString qsClassFilename = m_exp_param.class_dir().c_str() + QString("/part") + QString::number(pidx) + ".class";
  QString qsClassFilename = getClassFilename(pidx, bootstrap_type);

  cout << "saving classifier to " << qsClassFilename.toStdString() << endl;
  abc.saveClassifier(qsClassFilename.toStdString());
}


void PartApp::getScoreGridFileName(int imgidx, int pidx, bool flip,
                                   QString &qsFilename, QString &qsVarName) const
{
  QString qsScoreGridDir = m_exp_param.scoregrid_dir().c_str();

  qsFilename = qsScoreGridDir + "/imgidx" + QString::number(imgidx) + 
    "-pidx" + QString::number(pidx) + "-o" + QString::number((int)flip) + "-scoregrid.mat";
  //qsVarName = "cell_img_scoregrid";
  qsVarName = "cell_scoregrid";
}


/**
   this is a temporary implementation which loads template reponses produced by Ramanan's matlab code
 
   note: flip and bInterpolate are not used
 */
// void PartApp::loadScoreGrid(std::vector<std::vector<FloatGrid2> > &score_grid, int imgidx, int pidx, 
//                             bool flip, bool bInterpolate) const
// {
//   cout << "temporary PartApp::loadScoreGrid" << endl;

//   /* load score distribution parameters */

//   boost_math::double_vector mu_pos;
//   boost_math::double_vector sigma_pos;

//   boost_math::double_vector mu_neg;
//   boost_math::double_vector sigma_neg;

//   boost_math::double_vector upper_r;
//   boost_math::double_vector upper_r_x;

//   double prior_ratio;

//   //QString qsScoreDist = "/home/andriluk/EXPERIMENTS/ramanan/my_addons/score_dist_pr1.mat";
//   //QString qsScoreDist = "/home/andriluk/EXPERIMENTS/ramanan/my_addons/score_dist_pr10.mat"; // this worked the best 
//   //QString qsScoreDist = "/home/andriluk/EXPERIMENTS/ramanan/my_addons/score_dist_pr20.mat";
//   //QString qsScoreDist = "/home/andriluk/EXPERIMENTS/ramanan/my_addons/score_dist_pr40.mat";
//   QString qsScoreDist = "/home/andriluk/EXPERIMENTS/ramanan/my_addons/score_dist_pr80.mat";

//   cout << "qsScoreDist: " << qsScoreDist.toStdString() << endl;
  
//   assert(matlab_io::mat_load_double_vector(qsScoreDist, "mu_pos", mu_pos));
//   assert(matlab_io::mat_load_double_vector(qsScoreDist, "sigma_pos", sigma_pos));

//   assert(matlab_io::mat_load_double_vector(qsScoreDist, "mu_neg", mu_neg));
//   assert(matlab_io::mat_load_double_vector(qsScoreDist, "sigma_neg", sigma_neg));

//   assert(matlab_io::mat_load_double_vector(qsScoreDist, "upper_r", upper_r));
//   assert(matlab_io::mat_load_double_vector(qsScoreDist, "upper_r_x", upper_r_x));

//   assert(matlab_io::mat_load_double(qsScoreDist, "prior_ratio", prior_ratio));

//   for (int idx = 0; idx < (int)mu_pos.size(); ++idx) {
//     cout << "pidx: " << idx << ", mu_pos: " << mu_pos(idx) << ", mu_neg: " << mu_neg(idx) << 
//       ", sigma_pos: " << sigma_pos(idx) << ", sigma_neg: " << sigma_neg(idx) << endl;
//   }
//   cout << "prior_ratio: " << prior_ratio << endl;

//   /* load template responses */

//   QString qsFilename = "/home/andriluk/EXPERIMENTS/ramanan/new_download/save_part_resp_dir/";
//   qsFilename += "imgidx" + QString::number(imgidx) + "_pidx" + QString::number(pidx) + ".mat";

//   cout << "loading " << qsFilename.toStdString() << endl;

//   FloatGrid3 resp_h200 = matlab_io::mat_load_multi_array<FloatGrid3>(qsFilename, "resp_h200");

//   /** adjust the range of template responses */
//   //multi_array_op::multGrid(resp_h200, 0.05);

//   FloatGrid3 score_grid_pos = resp_h200;
//   multi_array_op::addGrid1(score_grid_pos, -mu_pos(pidx));
//   multi_array_op::multGrid2(score_grid_pos, score_grid_pos);
//   multi_array_op::multGrid(score_grid_pos, -1.0 / (2*square(sigma_pos(pidx))));
//   multi_array_op::computeExpGrid(score_grid_pos);
//   multi_array_op::multGrid(score_grid_pos, 1.0 / (sqrt(2*M_PI) * sigma_pos(pidx)));

//   FloatGrid3 score_grid_neg = resp_h200;
//   multi_array_op::addGrid1(score_grid_neg, -mu_neg(pidx));
//   multi_array_op::multGrid2(score_grid_neg, score_grid_neg);
//   multi_array_op::multGrid(score_grid_neg, -1.0 / (2*square(sigma_neg(pidx))));
//   multi_array_op::computeExpGrid(score_grid_neg);
//   multi_array_op::multGrid(score_grid_neg, 1.0 / (sqrt(2*M_PI) * sigma_neg(pidx)));

//   multi_array_op::multGrid(score_grid_neg, prior_ratio);
//   multi_array_op::addGrid2(score_grid_neg, score_grid_pos);

//   FloatGrid3 resp_h200_new = score_grid_pos;
//   multi_array_op::divGrid2(resp_h200_new, score_grid_neg);

//   int upper_clip_count = 0;

//   if (upper_r_x(pidx) > 0) {
//     float *pData = resp_h200.data();
//     float *pDataNew = resp_h200_new.data();
//     int nElements = resp_h200.num_elements();

//     for (int idx = 0; idx < nElements; ++idx) {
//       if (pData[idx] > upper_r_x(pidx)) {
//         pDataNew[idx] = upper_r(pidx);
//         ++upper_clip_count;
//       }
//     }
//   }

//   cout << "upper_clip_count: " << upper_clip_count << endl;
//   resp_h200 = resp_h200_new;

//   /* convert responses into vector/vector format */

//   int num_rot = resp_h200.shape()[0];
//   int img_height = resp_h200.shape()[1];
//   int img_width = resp_h200.shape()[2];

//   cout << "img_height: "  << img_height << 
//     ", img_width: " << img_width << 
//     ", num_rot: " << num_rot << endl;

//   score_grid.resize(1, std::vector<FloatGrid2>(num_rot, FloatGrid2(boost::extents[img_height][img_width])));

  
//   float min_val, max_val;
//   multi_array_op::getMinMax(resp_h200, min_val, max_val);
//   cout << "resp_h200, min_val: " << min_val << 
//     ", max_val: " << max_val << endl;

//   for (int ridx = 0; ridx < num_rot; ++ridx) {
//     //FloatGrid3View2 view_in =  resp_h200[boost::indices[index_range()][index_range()][ridx]];
//     //score_grid[0][ridx] = view_in;
//     score_grid[0][ridx] = resp_h200[ridx];
//   }
// }

/**
   load marginals for given part and scale

   dimensions of grid3: rotation, y, x

 */
FloatGrid3 PartApp::loadPartMarginal(int imgidx, int pidx, int scaleidx, bool flip) const 
{
  QString qsPartMarginalsDir = (m_exp_param.log_dir() + "/" + m_exp_param.log_subdir() + "/part_marginals").c_str();
  assert(filesys::check_dir(qsPartMarginalsDir));
  
  QString qsFilename = qsPartMarginalsDir + "/log_part_posterior_final" + 
    "_imgidx" + QString::number(imgidx) +
    "_scaleidx" + QString::number(scaleidx) +
    "_o" + QString::number((int)flip) + 
    "_pidx" + QString::number(pidx) + ".mat";

  cout << "loading " << qsFilename.toStdString() << endl;

  MATFile *f = matlab_io::mat_open(qsFilename, "r");
  assert(f != 0);

  /* make sure we use copy constructor here */ 
  FloatGrid3 _grid3 = matlab_io::mat_load_multi_array<FloatGrid3>(f, "log_prob_grid");
  matlab_io::mat_close(f);

  return _grid3;
}

void PartApp::loadScoreGrid(std::vector<std::vector<FloatGrid2> > &score_grid, int imgidx, int pidx, 
                            bool flip, bool bInterpolate) const
{
  cout << "PartApp::loadScoreGrid" << endl;

  QString qsFilename;
  QString qsVarName;
  getScoreGridFileName(imgidx, pidx, flip, qsFilename, qsVarName);

  cout << "loading scoregrid from " << qsFilename.toStdString() << endl;
  MATFile *f = matlab_io::mat_open(qsFilename, "r");
  assert(f != 0);
  matlab_io::mat_load_multi_array_vec2(f, qsVarName, score_grid);
  if (f != 0)
    matlab_io::mat_close(f);


  /** load transformation matrices */

  f = matlab_io::mat_open(qsFilename, "r"); assert(f != 0);
  FloatGrid4 transform_Ti2 = matlab_io::mat_load_multi_array<FloatGrid4>(f, "transform_Ti2");
  matlab_io::mat_close(f);

  f = matlab_io::mat_open(qsFilename, "r"); assert(f != 0);
  FloatGrid4 transform_T2g = matlab_io::mat_load_multi_array<FloatGrid4>(f, "transform_T2g");
  matlab_io::mat_close(f);

  /** find the image size, yes we need to load an image just for that :) */

  kma::ImageContent *kmaimg = kma::load_convert_gray_image(m_test_annolist[imgidx].imageName().c_str());
  assert(kmaimg != 0);
  int img_width = kmaimg->x();
  int img_height = kmaimg->y();
  delete kmaimg;

  /** map from the grid to image coordinates */

  int nScales = m_exp_param.num_scale_steps();
  int nRotations = m_exp_param.num_rotation_steps();

  assert((int)score_grid.size() == nScales);
  assert((int)score_grid[0].size() == nRotations);
    
  std::vector<std::vector<FloatGrid2> > score_grid_image(nScales, std::vector<FloatGrid2>());

  for (int scaleidx = 0; scaleidx < nScales; ++scaleidx) 
    for (int rotidx = 0; rotidx < nRotations; ++rotidx) {
      double_matrix T2g;
      double_matrix Ti2;

      multi_array_op::array_to_matrix(transform_T2g[boost::indices[scaleidx][rotidx][index_range()][index_range()]],
                                      T2g);
        
      multi_array_op::array_to_matrix(transform_Ti2[boost::indices[scaleidx][rotidx][index_range()][index_range()]],
                                      Ti2);

      double_matrix Tig = prod(Ti2, T2g);
        

      FloatGrid2 grid(boost::extents[img_height][img_width]);
      if (bInterpolate)
        multi_array_op::transform_grid_fixed_size(score_grid[scaleidx][rotidx], grid, Tig, 
                                                  part_detect::NO_CLASS_VALUE, TM_BILINEAR);
      else
        multi_array_op::transform_grid_fixed_size(score_grid[scaleidx][rotidx], grid, Tig, 
                                                  part_detect::NO_CLASS_VALUE, TM_DIRECT);

      score_grid_image[scaleidx].push_back(grid);
    }

  score_grid.clear();
  score_grid = score_grid_image;
}

void PartApp::saveScoreGrid(std::vector<std::vector<FloatGrid2> > &score_grid, int imgidx, int pidx, 
                            bool flip) const
{
  cout << "PartApp::savedScoreGrid" << endl;
  QString qsFilename;
  QString qsVarName;
  getScoreGridFileName(imgidx, pidx, flip, qsFilename, qsVarName);

  cout << "saving scoregrid to " << qsFilename.toStdString() << endl;
  MATFile *f = matlab_io::mat_open(qsFilename, "wz");
  assert(f != 0);
  matlab_io::mat_save_multi_array_vec2(f, qsVarName, score_grid);
  if (f != 0)
    matlab_io::mat_close(f);

}
