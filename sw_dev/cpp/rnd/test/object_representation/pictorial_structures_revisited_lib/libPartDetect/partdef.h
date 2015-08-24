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


#ifndef _PART_DEF_H_
#define _PART_DEF_H_

#include <QString>
#include <QPainter>

#include <libAnnotation/annotationlist.h>
#include <libAnnotation/annorect.h>

#include <libBoostMath/boost_math.h>

#include <libPartDetect/PartConfig.pb.h>
#include <libPartDetect/PartWindowParam.pb.h>

class QPolygonF;

/**
   structure that keeps data about part postition, orientation and bounding box
 */
struct PartBBox {

  PartBBox():part_pos(2), part_x_axis(2), part_y_axis(2) {}
  PartBBox(int ox, int oy, double xaxis_x, double xaxis_y, 
	   double _min_x, double _max_x, 
	   double _min_y, double _max_y);

  boost_math::double_vector part_pos;
  boost_math::double_vector part_x_axis;
  boost_math::double_vector part_y_axis;


  double max_proj_x;
  double min_proj_x;

  double max_proj_y;
  double min_proj_y;
};

/**
   compute position of part in the image (average over positions of annopoints)
 */
boost_math::double_vector get_part_position(const AnnoRect &annorect, const PartDef &partdef);

/**
   compute x axis of part coordinate system (in image coordinates)
 */
void get_part_x_axis(const AnnoRect &annorect, const PartDef &partdef, 
                     boost_math::double_vector &part_x_axis);

/**
   compute part bounding box (smallest rectangle that has the same orientation as part and contains all points
   which define part position)
 */
void get_part_bbox(const AnnoRect &annorect, const PartDef &partdef, 
                   PartBBox &part_bbox, double scale = 1.0);

void draw_bbox(QPainter &painter, const PartBBox &part_bbox, int coloridx = 0, int pen_width = 1);

/**
   visualize part positions and corresponding part bounding boxes, 
   visualization is saved in ./debug directory
 */
QImage visualize_parts(const PartConfig &conf, const PartWindowParam &window_param, const Annotation &annotation);

void get_part_polygon(PartBBox &part_bbox, QPolygonF &polygon); 

bool get_object_orientation(const AnnoRect &annorect);

bool annorect_has_part(const AnnoRect &annorect, const PartDef &partdef);

#endif
