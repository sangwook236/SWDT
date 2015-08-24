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

#include "FeatureGrid.h"

using std::vector;


FeatureGrid::FeatureGrid(int img_width, int img_height, 
                         boost_math::double_vector x_axis, boost_math::double_vector y_axis, 
                         double _desc_step, int _desc_size):desc_step(_desc_step), desc_size(_desc_size)
{
  assert(img_width > 0 && img_height > 0);
  assert(x_axis.size() == 2 && y_axis.size() == 2);

  rect.part_x_axis = x_axis;
  rect.part_y_axis = y_axis;

  rect.part_pos = boost_math::zero_double_vector(2);

  /* find bounding box in the rotated space */
  boost_math::double_matrix T12(2,2);
  column(T12, 0) = x_axis;
  column(T12, 1) = y_axis;
  boost_math::double_matrix T21(2, 2);
  boost_math::inv(T12, T21);

  boost_math::double_matrix corners = boost_math::zero_double_matrix(2, 4);
  corners(0, 0) = 0;              
  corners(1, 0) = 0;

  corners(0, 1) = img_width;
  corners(1, 1) = 0;

  corners(0, 2) = img_width;
  corners(1, 2) = img_height;

  corners(0, 3) = 0;
  corners(1, 3) = img_height;

  boost_math::double_matrix corners2 = prod(T21, corners);
  assert(corners2.size1() == 2 && corners2.size2() == 4);

  rect.min_proj_x = corners2(0, 0);
  rect.max_proj_x = corners2(0, 0);
  rect.min_proj_y = corners2(1, 0);
  rect.max_proj_y = corners2(1, 0);

  for (int i = 1; i < 4; ++i) {
    if (rect.min_proj_x > corners2(0, i))
      rect.min_proj_x = corners2(0, i);

    if (rect.max_proj_x < corners2(0, i))
      rect.max_proj_x = corners2(0, i);

    if (rect.min_proj_y > corners2(1, i))
      rect.min_proj_y = corners2(1, i);

    if (rect.max_proj_y < corners2(1, i))
      rect.max_proj_y = corners2(1, i);
  }

  /* same as in the other constructor */
  init();
}


FeatureGrid::FeatureGrid(const PartBBox &_rect, double _desc_step, int _desc_size):rect(_rect),
                                                                   desc_step(_desc_step),
                                                                   desc_size(_desc_size) 
{
  init();
}

void FeatureGrid::init()
{
  /* bounding box in grid coordinate system */
  double mx = rect.min_proj_x;
  double Mx = rect.max_proj_x;

  double my = rect.min_proj_y;
  double My = rect.max_proj_y;
    
  /* grid size */
  nx = (int)floor(((Mx - mx + 1) / desc_step)) + 1;
  ny = (int)floor(((My - my + 1) / desc_step)) + 1;

  desc.resize(ny, vector<vector<float> >(nx, vector<float>()));
}

bool FeatureGrid::concatenate(int grid_pos_x, int grid_pos_y, 
                              int grid_x_count, int grid_y_count, 
                              double grid_step, vector<float> &allfeatures)
{
  assert(grid_pos_x >= 0 && grid_pos_x + boost_math::round(grid_step*(grid_x_count - 1)) < nx);
  assert(grid_pos_y >= 0 && grid_pos_y + boost_math::round(grid_step*(grid_y_count - 1)) < ny);
  allfeatures.clear();

  // quickly test if all features are available
  for (int iy = 0; iy < grid_y_count; ++iy) 
    for (int ix = 0; ix < grid_x_count; ++ix) {

      int x = grid_pos_x + boost_math::round(ix*grid_step);
      int y = grid_pos_y + boost_math::round(iy*grid_step);
      if (desc[y][x].size() == 0)
        return false;
    }

  // concatenate feature vectors
  for (int iy = 0; iy < grid_y_count; ++iy) 
    for (int ix = 0; ix < grid_x_count; ++ix) {
      int x = grid_pos_x + boost_math::round(ix*grid_step);
      int y = grid_pos_y + boost_math::round(iy*grid_step);
      
      allfeatures.insert(allfeatures.begin(), desc[y][x].begin(), desc[y][x].end());
    }
  
  return true;
}

