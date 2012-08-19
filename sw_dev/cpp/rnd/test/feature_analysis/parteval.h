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

#include <libPartApp/partapp.h>

#include <libPartApp/ExpParam.pb.h>

#include <libPartDetect/PartWindowParam.pb.h>
#include <libPartDetect/partdef.h>

#include <libBoostMath/boost_math.h>

void get_bbox_endpoints(PartBBox &bbox, boost_math::double_vector &endpoint_top, 
	boost_math::double_vector &endpoint_bottom, double &seg_len);

void eval_segments(const PartApp &part_app, int firstidx, int lastidx);

bool is_gt_match(PartBBox &gt_bbox, PartBBox &detect_bbox, bool match_x_axis = false);

void draw_gt_circles(QPainter &painter, PartBBox &bbox);

void vis_backproject(const PartApp &part_app, int firstidx, int lastidx);
