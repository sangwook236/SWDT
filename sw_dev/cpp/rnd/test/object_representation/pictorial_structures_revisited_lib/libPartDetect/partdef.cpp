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
#include <fstream>
#include <climits>

#include <QString>
#include <QPointF>
#include <QImage>

#include <libBoostMath/boost_math.hpp>

#include "partdef.h"

using namespace std;

PartBBox::PartBBox(int ox, int oy, double xaxis_x, double xaxis_y, 
         double _min_x, double _max_x, 
         double _min_y, double _max_y) : part_pos(2), part_x_axis(2), part_y_axis(2), 
                                         max_proj_x(_max_x),
                                         min_proj_x(_min_x), 
                                         max_proj_y(_max_y), 
                                         min_proj_y(_min_y)
{ 
  part_pos(0) = ox;
  part_pos(1) = oy;
  part_x_axis(0) = xaxis_x;
  part_x_axis(1) = xaxis_y;
  part_y_axis(0) = -xaxis_y;
  part_y_axis(1) = xaxis_x;  
}

boost_math::double_vector get_part_position(const AnnoRect &annorect, const PartDef &partdef) 
{
  boost_math::double_vector v = boost_math::zero_double_vector(2);

  for (int i = 0; i < partdef.part_pos_size(); ++i) {
    uint id = partdef.part_pos(i);
    const AnnoPoint *p = annorect.get_annopoint_by_id(id);
    assert(p != NULL);

    /* make sure we get the right annopoint */
    assert(p->id == (int)id);

    v(0) += p->x;
    v(1) += p->y;
  }
  v *= 1.0/partdef.part_pos_size();

  return v;
}

void get_part_x_axis(const AnnoRect &annorect, const PartDef &partdef, 
                     boost_math::double_vector &part_x_axis) 
{
  assert(part_x_axis.size() == 2);
  if (partdef.has_part_x_axis_from() && partdef.has_part_x_axis_to()) {
    uint fromid = partdef.part_x_axis_from();
    uint toid = partdef.part_x_axis_to();

    const AnnoPoint *from = annorect.get_annopoint_by_id(fromid);
    const AnnoPoint *to = annorect.get_annopoint_by_id(toid);

    /* make sure we get the right annopoint */
    assert(from != NULL && to != NULL);

    part_x_axis(0) = to->x - from->x;
    part_x_axis(1) = to->y - from->y;

    boost_math::double_matrix R = boost_math::get_rotation_matrix(partdef.part_x_axis_offset()*M_PI/180.0);
    part_x_axis = prod(R, part_x_axis);
  }
  else {
    part_x_axis(0) = 1;
    part_x_axis(1) = 0;
  }
}

void get_part_bbox(const AnnoRect &annorect, const PartDef &partdef, 
                   PartBBox &part_bbox, double scale)
{
  assert(annorect_has_part(annorect, partdef));

  bool flip = get_object_orientation(annorect);

  assert(!flip);

  part_bbox.part_pos = get_part_position(annorect, partdef);

  get_part_x_axis(annorect, partdef, part_bbox.part_x_axis);
  part_bbox.part_x_axis /= norm_2(part_bbox.part_x_axis);

  part_bbox.part_y_axis = prod(boost_math::get_rotation_matrix(M_PI/2), part_bbox.part_x_axis);
  
  part_bbox.max_proj_x = -numeric_limits<double>::infinity();
  part_bbox.min_proj_x = numeric_limits<double>::infinity();

  part_bbox.max_proj_y = -numeric_limits<double>::infinity();
  part_bbox.min_proj_y = numeric_limits<double>::infinity();

  // part bounding box should include all points used to compute part position
  for (int i = 0; i < partdef.part_pos_size(); ++i) {
    
    uint id = partdef.part_pos(i);
    const AnnoPoint *p = annorect.get_annopoint_by_id(id);

    assert(p != NULL);
    assert(p->id == (int)id);

    boost_math::double_vector annopoint(2);

    annopoint(0) = p->x;
    annopoint(1) = p->y;

    annopoint = annopoint - part_bbox.part_pos;
    
    double proj_x = inner_prod(part_bbox.part_x_axis, annopoint);
    double proj_y = inner_prod(part_bbox.part_y_axis, annopoint);

    if (proj_x < part_bbox.min_proj_x)
      part_bbox.min_proj_x = proj_x;
    if (proj_x > part_bbox.max_proj_x)
      part_bbox.max_proj_x = proj_x;

    if (proj_y < part_bbox.min_proj_y)
      part_bbox.min_proj_y = proj_y;
    if (proj_y > part_bbox.max_proj_y)
      part_bbox.max_proj_y = proj_y;
  }// annopoints

  part_bbox.min_proj_x -= scale*partdef.ext_x_neg();
  part_bbox.max_proj_x += scale*partdef.ext_x_pos();

  part_bbox.min_proj_y -= scale*partdef.ext_y_neg();
  part_bbox.max_proj_y += scale*partdef.ext_y_pos();
}

void get_part_polygon(const PartBBox &part_bbox, QPolygonF &polygon)
{
  polygon.clear();

  boost_math::double_vector t(2);
  t = part_bbox.part_x_axis*part_bbox.min_proj_x + part_bbox.part_y_axis*part_bbox.min_proj_y + part_bbox.part_pos;
  polygon.push_back(QPointF(t(0), t(1)));
    
  t = part_bbox.part_x_axis*part_bbox.max_proj_x + part_bbox.part_y_axis*part_bbox.min_proj_y + part_bbox.part_pos;
  polygon.push_back(QPointF(t(0), t(1)));

  t = part_bbox.part_x_axis*part_bbox.max_proj_x + part_bbox.part_y_axis*part_bbox.max_proj_y + part_bbox.part_pos;
  polygon.push_back(QPointF(t(0), t(1)));

  t = part_bbox.part_x_axis*part_bbox.min_proj_x + part_bbox.part_y_axis*part_bbox.max_proj_y + part_bbox.part_pos;
  polygon.push_back(QPointF(t(0), t(1)));
}

/** 
    coloridx: 0 - yellow, 1 - red, if > 1 - green; if < 0 - only position without bounding box is drawn
 */
void draw_bbox(QPainter &painter, const PartBBox &part_bbox, int coloridx, int pen_width)
{    
  painter.setPen(Qt::yellow);
  
  int marker_radius = 3;
  int part_axis_length = 10;

  painter.drawEllipse(QRect((int)(part_bbox.part_pos(0) - marker_radius), (int)(part_bbox.part_pos(1) - marker_radius), 
                            2*marker_radius, 2*marker_radius));

  boost_math::double_vector v(2);
  v = part_bbox.part_pos + part_axis_length * part_bbox.part_x_axis;
  painter.drawLine((int)part_bbox.part_pos(0), (int)part_bbox.part_pos(1), (int)v(0), (int)v(1));

  painter.setPen(Qt::red);
  v = part_bbox.part_pos + part_axis_length * part_bbox.part_y_axis;
  painter.drawLine((int)part_bbox.part_pos(0), (int)part_bbox.part_pos(1), (int)v(0), (int)v(1));
  painter.setPen(Qt::yellow);

  if (coloridx >= 0) {
    QPen pen;

    if (coloridx == 0) 
      pen.setColor(Qt::yellow);
    else if (coloridx == 1)
      pen.setColor(Qt::red);
    else
      pen.setColor(Qt::green);

    pen.setJoinStyle(Qt::RoundJoin);
    pen.setWidth(pen_width);

    painter.setPen(pen);

    QPolygonF polygon;
    get_part_polygon(part_bbox, polygon);
    painter.drawPolygon(polygon);
  }

}

QImage visualize_parts(const PartConfig &conf, const PartWindowParam &window_param, const Annotation &annotation)
{
  assert(annotation.size() > 0);

  double scale = (annotation[0].bottom() - annotation[0].top())/window_param.train_object_height();
  cout << "visualize_parts, scale: " << scale << endl;

  QImage _img;
  cout << "loading image" << endl;
  assert(_img.load(annotation.imageName().c_str()));
  QImage img = _img.convertToFormat(QImage::Format_RGB32);

  QPainter painter(&img);  
  for (int pidx = 0; pidx < conf.part_size(); ++pidx) {
    PartBBox part_bbox;
    get_part_bbox(annotation[0], conf.part(pidx), part_bbox, scale);

    int coloridx = 1;
    int pen_width = 2;

    if (conf.part(pidx).is_detect())
      draw_bbox(painter, part_bbox, coloridx, pen_width);
    else
      draw_bbox(painter, part_bbox, -1); // only draw center point and axis (skip bounding box)

  }

  cout << "done" << endl;
  return img;
}

bool get_object_orientation(const AnnoRect &annorect)
{
  if (annorect.m_vAnnoPoints.size() == 11) {   /* pedestrians */
    if (annorect.m_vAnnoPoints[0].x < annorect.m_vAnnoPoints[1].x)
      return false;
    else
      return true;
  }
  else if (annorect.m_vAnnoPoints.size() == 12) { /* buffy */
    return false;
  }
  else if (annorect.m_vAnnoPoints.size() == 16) { /* ramanan */
    return false;
  }
  else {
    assert(false && "unknown annopoint configuration");
  }

}

bool annorect_has_part(const AnnoRect &annorect, const PartDef &partdef)
{
  bool bres = true;

  for (int idx = 0; idx < partdef.part_pos_size(); ++idx) {
    uint apidx = partdef.part_pos(idx);

    if (annorect.get_annopoint_by_id(apidx) == NULL) {
      bres = false;
      break;
    }
  }

  return bres;
}

