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

#include <libAdaBoost/AdaBoost.h>

#include <libKMA2/gauss_iir/gauss_iir.h>
#include <libKMA2/kmaimagecontent.h>
#include <libKMA2/ShapeDescriptor.h>
#include <libKMA2/descriptor/feature.h>

#include <libBoostMath/boost_math.h>
#include <libBoostMath/boost_math.hpp>
#include <libBoostMath/homogeneous_coord.h>

#include <libMisc/misc.hpp>
#include <libFilesystemAux/filesystem_aux.h>
#include <libMatlabIO/matlab_io.h>

#include "partdetect.h"
#include "partdef.h"

using boost_math::double_matrix;
using boost_math::double_vector;

using namespace std;

namespace part_detect { 

  /**
     compute desriptors on the uniform grid 

     rect - define position, orientation and size of grid in the image

     desc_step - distance between descriptors
     desc_size - descriptor size
  */
  void computeDescriptorGrid(kma::ImageContent *input_image, FeatureGrid &grid, QString qsDESC)
  {
    initPatchMask(PATCH_SIZE);

    /* descriptor patch */
    DARY *patch = new DARY(PATCH_SIZE, PATCH_SIZE);

    /* descritor */
    int _ShapeSize = kma::shape::SrSize * kma::shape::ScSize * kma::shape::SOriSize;
    vector<float> ds(_ShapeSize);

    double patch_scale = (2*grid.desc_size + 1) / (float)PATCH_SIZE;

    DARY *source_image = input_image;
    DARY *smooth_input_image = NULL;

    /* smooth once for all descriptors if patch < descriptor */
    if (patch_scale > 1.0) {
      smooth_input_image = new DARY(input_image->y(), input_image->x());
      smooth(input_image, smooth_input_image, patch_scale);
      source_image = smooth_input_image;
    }

    boost_math::double_matrix patch_lcpos;
    boost_math::double_matrix patch_lrpos;

    assert(qsDESC == "SHAPEv5" && "unknown descriptor type");

    kma::precompute_patch_idx(patch_lcpos, patch_lrpos, (int)floor(PATCH_SIZE/2.0), kma::shape::SrSize, kma::shape::ScSize);

    assert((int)grid.desc.size() == grid.ny);

    for (int iy = 0; iy < grid.ny; ++iy) { 
      //assert((int)grid.desc[iy].size() == grid.nx);

      if (!((int)grid.desc[iy].size() == grid.nx)) {
        cout << "iy: " << iy << endl;
        cout << "required size: " << grid.nx << endl;
        cout << "current size: " << grid.desc[iy].size() << endl;
        assert(false);
      }

      for (int ix = 0; ix < grid.nx; ++ix) {
        ds.assign(ds.size(), 0.0);
        grid.desc[iy][ix].clear();

        /* grid coordinates */
        double x2 = grid.rect.min_proj_x + ix*grid.desc_step;
        double y2 = grid.rect.min_proj_y + iy*grid.desc_step;

        /* image coordinates */
        boost_math::double_vector imgpos = grid.origin() + x2*grid.x_axis() + y2*grid.y_axis();

        /* test that descriptor is inside the image */
        if (imgpos(0) >= grid.desc_size + 1 && imgpos(0) <= source_image->x() - (grid.desc_size + 1) && 
            imgpos(1) >= grid.desc_size + 1 && imgpos(1) <= source_image->y() - (grid.desc_size + 1)) {

          /* map image region to patch */
          patch->interpolate(source_image, imgpos(0), imgpos(1), 
                             patch_scale*grid.x_axis()(0), patch_scale*grid.y_axis()(0), 
                             patch_scale*grid.x_axis()(1), patch_scale*grid.y_axis()(1));

          /* mean/var normalization, rescale with x = 128 + 50*x, clip to [0, 255] (last 3 parameters are not important) */
          normalize(patch, 0, 0, 0);
          //patch->writePNG(qsFilename.toStdString().c_str());

          assert(patch_lcpos.size1() > 0 && patch_lcpos.size2() > 0);

          /* compute SHAPEv5 */
          kma::KMAcomputeShape(patch_lcpos, patch_lrpos, patch, ds);

          grid.desc[iy][ix].insert(grid.desc[iy][ix].end(), ds.begin(), ds.end());
        }
      }
    }// grid positions

    delete smooth_input_image;
    delete patch;    
  }

  int get_app_group_idx(const PartConfig &part_conf, int pid)
  {
    for (int agidx = 0; agidx < part_conf.app_group_size(); ++agidx) 
      for (int idx = 0; idx < part_conf.app_group(agidx).part_id_size(); ++idx) {
        if (part_conf.app_group(agidx).part_id(idx) == pid)
          return agidx;
      }

    return -1;
  }

  int pidx_from_pid(const PartConfig &part_conf, int pid)
  {
    for (int pidx = 0; pidx < part_conf.part_size(); ++pidx) {
      assert(part_conf.part(pidx).has_part_id());

      if (part_conf.part(pidx).part_id() == pid)
        return pidx;
    }

    assert(false && "part id not found");
    return -1;
  }

  void get_window_feature_counts(const AbcDetectorParam &abc_param, const PartWindowParam::PartParam &part_window_param, 
                                 int &grid_x_count, int &grid_y_count)
  {
    grid_x_count = part_window_param.window_size_x() / abc_param.desc_step() + 1;
    grid_y_count = part_window_param.window_size_y() / abc_param.desc_step() + 1;
  }

  /**
     compute the average size of part bounding box
  */
  void compute_part_window_param(const AnnotationList &annolist, const PartConfig &partconf, PartWindowParam &windowparam)
  {
    int nParts = partconf.part_size();
    int nImages = annolist.size();

    /** 
        compute reference object height, use only rectangles that have all parts 
        annotated (rectangles with missing parts are sometimes smaller then reference object height)
    */
    double train_object_height = 0;
    int n = 0;
    for (int imgidx = 0; imgidx < nImages; ++imgidx) {
      int nRects = annolist[imgidx].size(); 
      for (int ridx = 0; ridx < nRects; ++ridx) {

        bool bHasAllParts = true;

        //for (int pidx = 0; pidx < partconf.part_size(); ++pidx) 
        /** 
            
        TEMPORARY !!!!  Needed for experiments on HumanEva when training is done on the combined set of images
        from Buffy, Ramanan, and TUD Pedestrians

         */
//         assert(partconf.part_size() == 10);
//         for (int pidx = 0; pidx < 6; ++pidx) 
//           if (!annorect_has_part(annolist[imgidx][ridx], partconf.part(pidx))) {
//             bHasAllParts = false;
//             break;
//           }

        if (bHasAllParts) {
          train_object_height += abs(annolist[imgidx][ridx].bottom() - annolist[imgidx][ridx].top());
          ++n;
        }

      }// rectangles
    }// images

    assert(n > 0);
    
    train_object_height /= n;
    windowparam.set_train_object_height(train_object_height);
    
    cout << "train_object_height: " << train_object_height << endl;
  
    /** determine average dimensions of the parts */
    windowparam.clear_part();
    for (int pidx = 0; pidx < nParts; ++pidx) {
      PartWindowParam::PartParam *pPartParam = windowparam.add_part();
      pPartParam->set_part_id(partconf.part(pidx).part_id());
      pPartParam->set_window_size_x(0);
      pPartParam->set_window_size_y(0);
      pPartParam->set_pos_offset_x(0);
      pPartParam->set_pos_offset_y(0);

      /** begin debug */
      cout << "part_pos_size:" << partconf.part(pidx).part_pos_size() << endl;
      for (int idx = 0; idx < partconf.part(pidx).part_pos_size(); ++idx)
        cout << "\tapidx: " << partconf.part(pidx).part_pos(idx) << endl;
      /** end debug */

      if (partconf.part(pidx).is_detect()) {
        double sum_window_size_x = 0;
        double sum_window_size_y = 0;
        double sum_pos_offset_x = 0;
        double sum_pos_offset_y = 0;
    
        int nAnnoRects = 0;
        for (int imgidx = 0; imgidx < nImages; ++imgidx) {
          int nRects = annolist[imgidx].size(); 

          for (int ridx = 0; ridx < nRects; ++ridx) {

//             cout << "pidx: " << pidx << 
//               " imgidx: " << imgidx << 
//               ", ridx: " << ridx << 
//               ", has_part: " << annorect_has_part(annolist[imgidx][ridx], partconf.part(pidx)) << endl;

            if (annorect_has_part(annolist[imgidx][ridx], partconf.part(pidx))) {
              PartBBox bbox;
              get_part_bbox(annolist[imgidx][ridx], partconf.part(pidx), bbox);

              sum_window_size_x += (bbox.max_proj_x - bbox.min_proj_x);
              sum_window_size_y += (bbox.max_proj_y - bbox.min_proj_y);
              sum_pos_offset_x += bbox.min_proj_x;
              sum_pos_offset_y += bbox.min_proj_y;

              ++nAnnoRects;
            }
          }// rects

        }// images

        assert(nAnnoRects > 0);
        cout << "processed rects: " << nAnnoRects << endl;

        pPartParam->set_window_size_x((int)(sum_window_size_x/nAnnoRects));
        pPartParam->set_window_size_y((int)(sum_window_size_y/nAnnoRects));

        /* average offset of part position with respect to top/left corner */
        pPartParam->set_pos_offset_x((int)(-sum_pos_offset_x/nAnnoRects));
        pPartParam->set_pos_offset_y((int)(-sum_pos_offset_y/nAnnoRects));
      }

    }// parts

    /** compute offset of the root part with respect to bounding box center */

    int rootpart_idx = -1;

    for (int pidx = 0; pidx < nParts; ++pidx) {

      if (partconf.part(pidx).is_root()) {
        rootpart_idx = pidx;
        break;
      }
    }

    assert(rootpart_idx >=0 && "missing root part");
    double_vector bbox_offset = boost_math::double_zero_vector(2);

    int total_rects = 0;
    for (int imgidx = 0; imgidx < nImages; ++imgidx) {

      for (uint ridx = 0; ridx < annolist[imgidx].size(); ++ridx) {
        bool bHasAllParts = true;

        //for (int pidx = 0; pidx < partconf.part_size(); ++pidx) 
        // TEMPORARY !!!! needed for training of HumanEva on combined training set
//         assert(partconf.part_size() == 10);
//         for (int pidx = 0; pidx < 6; ++pidx) 
//           if (!annorect_has_part(annolist[imgidx][ridx], partconf.part(pidx))) {
//             bHasAllParts = false;
//             break;
//           }

        //if (annorect_has_part(annolist[imgidx][ridx], partconf.part(rootpart_idx))) {
        if (bHasAllParts) {
          PartBBox bbox;
          get_part_bbox(annolist[imgidx][ridx], partconf.part(rootpart_idx), bbox);
      
          double_vector bbox_center(2);
          bbox_center(0) = 0.5*(annolist[imgidx][ridx].left() + annolist[imgidx][ridx].right());
          bbox_center(1) = 0.5*(annolist[imgidx][ridx].top() + annolist[imgidx][ridx].bottom());

          bbox_offset += bbox_center - bbox.part_pos;
          ++total_rects;
        }
      }
    }
    bbox_offset /= total_rects;

    windowparam.set_bbox_offset_x(bbox_offset(0));
    windowparam.set_bbox_offset_y(bbox_offset(1));
  }


  void sample_random_partrect(boost::variate_generator<boost::mt19937, boost::uniform_real<> > &gen_01,
                              double min_scale, double max_scale,
                              double min_rot, double max_rot, 
                              double min_pos_x, double max_pos_x,
                              double min_pos_y, double max_pos_y,
                              int rect_width, int rect_height, 
                              PartBBox &partrect, double &scale, double &rot)
  {
    scale = min_scale + (max_scale - min_scale)*gen_01();
    rot = min_rot + (max_rot - min_rot)*gen_01();

    partrect.part_pos(0) = min_pos_x + (max_pos_x - min_pos_x)*gen_01();
    partrect.part_pos(1) = min_pos_y + (max_pos_y - min_pos_y)*gen_01();

    //cout << alpha << ", " << scale << ", " << partrect.part_pos(0) <<  ", " << partrect.part_pos(1) << endl;

    partrect.part_x_axis(0) = cos(rot);
    partrect.part_x_axis(1) = sin(rot);

    partrect.part_y_axis(0) = -partrect.part_x_axis(1);
    partrect.part_y_axis(1) = partrect.part_x_axis(0);

    rect_width = (int)(rect_width*scale);
    rect_height = (int)(rect_height*scale);

    partrect.min_proj_x = -floor(rect_width/2.0);
    partrect.max_proj_x = partrect.min_proj_x + rect_width - 1;
  
    partrect.min_proj_y = -floor(rect_height/2.0);
    partrect.max_proj_y = partrect.min_proj_y + rect_height - 1;
  }

}// namespace 
