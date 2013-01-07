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

#include <libFilesystemAux/filesystem_aux.h>
#include <libMultiArray/multi_array_def.h>

#include <libPartApp/partapp_aux.hpp>

#define CHAR16_T  //-- [] 2012/08/08: Sang-Wook Lee
#include <libMatlabIO/matlab_io.hpp>
#include <libMisc/misc.hpp>

#include <libBoostMath/homogeneous_coord.h>

#include <libMultiArray/multi_array_op.hpp>

#include "parteval.h"

using namespace std;

using boost_math::double_vector;

void get_bbox_endpoints(PartBBox &bbox, double_vector &endpoint_top, double_vector &endpoint_bottom, double &seg_len)
{
	seg_len = bbox.max_proj_y - bbox.min_proj_y;

	endpoint_top = bbox.part_pos + bbox.min_proj_y*bbox.part_y_axis;
	endpoint_bottom = bbox.part_pos + bbox.max_proj_y*bbox.part_y_axis;
}

void get_bbox_endpoints_xaxis(PartBBox &bbox, double_vector &endpoint_left, double_vector &endpoint_right, double &seg_len)
{
	seg_len = bbox.max_proj_x - bbox.min_proj_x;

	endpoint_left = bbox.part_pos + bbox.min_proj_x*bbox.part_x_axis;
	endpoint_right = bbox.part_pos + bbox.max_proj_x*bbox.part_x_axis;
}

bool is_gt_match(PartBBox &gt_bbox, PartBBox &detect_bbox, bool match_x_axis) 
{
	cout << "is_gt_match, match_x_axis: " << match_x_axis << endl;

	if (!match_x_axis) {
		double gt_seg_len;
		double_vector gt_endpoint_top;
		double_vector gt_endpoint_bottom;
		get_bbox_endpoints(gt_bbox, gt_endpoint_top, gt_endpoint_bottom, gt_seg_len);

		double detect_seg_len;
		double_vector detect_endpoint_top;
		double_vector detect_endpoint_bottom;
		get_bbox_endpoints(detect_bbox, detect_endpoint_top, detect_endpoint_bottom, detect_seg_len);

		bool match_top = (ublas::norm_2(gt_endpoint_top - detect_endpoint_top) < 0.5*gt_seg_len);
		bool match_bottom = (ublas::norm_2(gt_endpoint_bottom - detect_endpoint_bottom) < 0.5*gt_seg_len);
		return match_top && match_bottom;
	}
	else {
		double gt_seg_len;
		double_vector gt_endpoint_left;
		double_vector gt_endpoint_right;
		get_bbox_endpoints_xaxis(gt_bbox, gt_endpoint_left, gt_endpoint_right, gt_seg_len);

		double detect_seg_len;
		double_vector detect_endpoint_left;
		double_vector detect_endpoint_right;
		get_bbox_endpoints_xaxis(detect_bbox, detect_endpoint_left, detect_endpoint_right, detect_seg_len);

		bool match_left = (ublas::norm_2(gt_endpoint_left - detect_endpoint_left) < 0.5*gt_seg_len);
		bool match_right = (ublas::norm_2(gt_endpoint_right - detect_endpoint_right) < 0.5*gt_seg_len);
		return match_left && match_right;
	}
}

/**
note: there are several peculiarities when evaluating TUD Pedestrians
- shrinking of bounding boxes should be turned off
- endpoints of foot part are along the x axis, not along the y axis 

currently all this things are HARDCODED
*/

void eval_segments(const PartApp &part_app, int firstidx, int lastidx)
{
	bool bEvalUnaries = false;

	int seg_total = 0;
	int seg_correct = 0;

	int nParts = part_app.m_part_conf.part_size();

	cout << "nParts: " << nParts << endl;

	vector<int> per_segment_correct(nParts, 0);

	QString qsSegEvalImagesDir = (part_app.m_exp_param.log_dir() + "/" + part_app.m_exp_param.log_subdir() + "/part_marginals/seg_eval_images").c_str();

	if (!filesys::check_dir(qsSegEvalImagesDir))
		filesys::create_dir(qsSegEvalImagesDir);

	for (int imgidx = firstidx; imgidx <= lastidx; ++imgidx) {      

		int scaleidx = 0;
		double scale = scale_from_index(part_app.m_exp_param, scaleidx);

		QImage _img;
		assert(_img.load(part_app.m_test_annolist[imgidx].imageName().c_str()));
		QImage img = _img.convertToFormat(QImage::Format_RGB32);
		QPainter painter(&img);
		painter.setRenderHints(QPainter::Antialiasing);

		int num_correct_cur_imgidx = 0;

		for (int pidx = 0; pidx < part_app.m_part_conf.part_size(); ++pidx) {

			int max_ridx= -1;
			int max_ix = -1, max_iy = -1;

			PartBBox bbox;

			if (bEvalUnaries) {

				/** this only works for ramanan's and buffy single scale setting */
				assert(part_app.m_part_conf.part_size() == 10 ||
					part_app.m_part_conf.part_size() == 6);

				vector<vector<FloatGrid2> > part_detections;

				bool flip = false;
				bool bInterpolate = false;

				part_app.loadScoreGrid(part_detections, imgidx, pidx, flip, bInterpolate);

				assert(part_detections.size() == 1);

				max_ridx = 0;
				max_iy = 0;
				max_ix = 0;
				float max_val = part_detections[0][0][0][0];

				int nRotations = part_app.m_exp_param.num_rotation_steps();
				assert((int)part_detections[0].size() == nRotations);
				int img_height = part_detections[0][0].shape()[0];
				int img_width = part_detections[0][0].shape()[1];

				cout << "img_width: " << img_width << ", img_height: " << img_height << endl;

				for (int ridx = 0; ridx < nRotations; ++ridx)
					for (int ix = 0; ix < img_width; ++ix)
						for (int iy = 0; iy < img_height; ++iy) {
							if (part_detections[0][ridx][iy][ix] > max_val) {
								max_ridx = ridx;
								max_ix = ix;
								max_iy = iy;
								max_val = part_detections[0][ridx][iy][ix];
							}
						}          
			}
			else {
				bool flip = false;
				FloatGrid3 grid3 = part_app.loadPartMarginal(imgidx, pidx, scaleidx, flip);

				max_ridx = 0;
				max_iy = 0;
				max_ix = 0;
				float max_val = grid3[0][0][0];

				int nRotations = part_app.m_exp_param.num_rotation_steps();
				int img_height = grid3.shape()[1];
				int img_width = grid3.shape()[2];

				for (int ridx = 0; ridx < nRotations; ++ridx)
					for (int ix = 0; ix < img_width; ++ix)
						for (int iy = 0; iy < img_height; ++iy) {
							if (grid3[ridx][iy][ix] > max_val) {
								max_ridx = ridx;
								max_ix = ix;
								max_iy = iy;
								max_val = grid3[ridx][iy][ix];
							}

						}          
			}

			cout << "pidx: " << pidx << 
				", max_ridx: " << max_ridx << 
				", max_ix: " << max_ix << 
				", max_iy: " << max_iy << endl;

			bbox_from_pos(part_app.m_exp_param, part_app.m_window_param.part(pidx), 
				scaleidx, max_ridx, max_ix, max_iy, 
				bbox);

			/** 
			remove part offset in y dir 
			shrink box in x direction
			*/

			bool no_y_offset = true;
			bool shrink_x_offset = true;

			if (no_y_offset) {
				bbox.min_proj_y += scale * part_app.m_part_conf.part(pidx).ext_y_neg();
				bbox.max_proj_y -= scale * part_app.m_part_conf.part(pidx).ext_y_pos();
			}

			if (shrink_x_offset) {
				bbox.max_proj_x *= 0.7;
				bbox.min_proj_x *= 0.7;
			}

			assert(part_app.m_test_annolist[imgidx].size() == 1);
			bool match = false;


			if (part_app.m_test_annolist[imgidx][0].m_vAnnoPoints.size() > 0) {

				PartBBox gt_bbox;
				assert(part_app.m_test_annolist[imgidx].size() == 1);
				get_part_bbox(part_app.m_test_annolist[imgidx][0], part_app.m_part_conf.part(pidx), gt_bbox, scale);

				//draw_gt_circles(painter, gt_bbox);

				bool match_x_axis = false;
				match = is_gt_match(gt_bbox, bbox, match_x_axis);

				if (match) {
					++seg_correct;
					per_segment_correct[pidx]++;
				}

				++seg_total;
			}
			else {
				cout << "missing annopoints in image: " << imgidx << ", skipping part evaluation " << endl; 
			}

			int coloridx = 1;
			int pen_width = 2;
			draw_bbox(painter, bbox, coloridx, pen_width);

			cout << "pidx: " << pidx << ", match: " << (int)match  << endl;

			num_correct_cur_imgidx += (int)match;

		}// parts

		cerr << imgidx << " " << num_correct_cur_imgidx << endl;

		QString qsFilename2 = qsSegEvalImagesDir + "/img_" + padZeros(QString::number(imgidx), 3) + ".png";
		cout << "saving " << qsFilename2.toStdString() << endl;
		assert(img.save(qsFilename2));

	}// images

	cout << "seg_correct: " << seg_correct << endl;
	cout << "seg_total: " << seg_total << endl;

	if (seg_total != 0) {
		cout << "ratio: " << seg_correct / (double)seg_total << endl;

		int per_segment_total = seg_total / nParts;

		cout << endl;
		for (int pidx = 0; pidx < nParts; ++pidx) {
			cout << "part: " << pidx << 
				", correct: " << per_segment_correct[pidx] << 
				", total: " << per_segment_total << 
				", ratio: " << per_segment_correct[pidx]/(double)per_segment_total << endl;
		}
	}


}

void draw_gt_circles(QPainter &painter, PartBBox &bbox)
{
	double seg_len;
	double_vector endpoint_top;
	double_vector endpoint_bottom;

	get_bbox_endpoints(bbox, endpoint_top, endpoint_bottom, seg_len);

	painter.setPen(Qt::yellow);
	painter.drawEllipse((int)(endpoint_top(0) - 0.5*seg_len), (int)(endpoint_top(1) - 0.5*seg_len), 
		(int)seg_len, (int)seg_len);
	painter.drawEllipse((int)(endpoint_bottom(0) - 0.5*seg_len), (int)(endpoint_bottom(1) - 0.5*seg_len), 
		(int)seg_len, (int)seg_len);
}

void vis_backproject(const PartApp &part_app, int firstidx, int lastidx)
{

	cout << "vis_backproject" << endl;
	cout << "processing images from " << firstidx << " to " << lastidx << endl;

	bool bEvalLoopyLoopy = false;

	/** load samples, load samples ordering */
	QString qsReoderSamplesDir;
	QString qsBackprojectDir;

	if (bEvalLoopyLoopy) {
		qsReoderSamplesDir = (part_app.m_exp_param.log_dir() + "/" + 
			part_app.m_exp_param.log_subdir() + "/loopy_reordered_samples").c_str();

		qsBackprojectDir = (part_app.m_exp_param.log_dir() + "/" + 
			part_app.m_exp_param.log_subdir() + "/loopy_backproject").c_str();

	}
	else {
		qsReoderSamplesDir = (part_app.m_exp_param.log_dir() + "/" + 
			part_app.m_exp_param.log_subdir() + "/tree_reordered_samples").c_str();      

		qsBackprojectDir = (part_app.m_exp_param.log_dir() + "/" + 
			part_app.m_exp_param.log_subdir() + "/tree_backproject").c_str();

	}

	cout << "qsReoderSamplesDir: " << qsReoderSamplesDir.toStdString() << endl;
	assert(filesys::check_dir(qsReoderSamplesDir));

	if (!filesys::check_dir(qsBackprojectDir)) 
		assert(filesys::create_dir(qsBackprojectDir));

	cout << "storing images in " << qsBackprojectDir.toStdString() << endl;

	/** load original annolist */

	AnnotationList annolist_original;
	annolist_original.load("/home/andriluk/sandbox/pm2/partdef/log_dir/exp_param_parts10_shape5_train400_peopleonly/results/exp_param_parts10_shape5_train400_peopleonly_spmnone-nms.al");

	assert(annolist_original.size() == part_app.m_test_annolist.size());

	/** load samples */

	for (int imgidx = firstidx; imgidx <= lastidx; ++imgidx) {
		QString qsReoderSamplesFile = qsReoderSamplesDir + "/loopy_reorder_imgidx" + padZeros(QString::number(imgidx), 4) + ".mat";   
		assert(filesys::check_file(qsReoderSamplesFile));

		cout << "loading samples order from " << qsReoderSamplesFile.toStdString() << endl;
		FloatGrid2 sorted_sampidx = matlab_io::mat_load_multi_array<FloatGrid2>(qsReoderSamplesFile, "sorted_sampidx");

		QString qsSamplesDir = (part_app.m_exp_param.log_dir() + "/" + part_app.m_exp_param.log_subdir() + "/samples").c_str();
		assert(filesys::check_dir(qsSamplesDir) && "samples dir not found");
		QString qsFilename = qsSamplesDir + "/samples" + 
			"_imgidx" + QString::number(imgidx) +
			"_scaleidx" + QString::number(0) + 
			"_o" + QString::number((int)false) + ".mat";
		assert(filesys::check_file(qsFilename));

		cout << "loading samples from " << qsFilename.toStdString() << endl;
		FloatGrid3 all_samples = matlab_io::mat_load_multi_array<FloatGrid3>(qsFilename, "all_samples");

		cout << "loaded " << all_samples.shape()[0] << " samples" << endl;

		int sampidx = (int)sorted_sampidx[sorted_sampidx.shape()[0] - 1][0];
		//int sampidx = 0;

		double sampidx_lik = sorted_sampidx[sorted_sampidx.shape()[0] - 1][1];
		cout << "sampidx: " << sampidx << ", sampidx_lik: " << sampidx_lik << endl;

		/** load the transformation */
		QString qsOriginalImg = annolist_original[imgidx].imageName().c_str();
		QString qsProjImg = part_app.m_test_annolist[imgidx].imageName().c_str();

		QString qsTransfDir, qsTmp;
		filesys::split_filename(qsProjImg, qsTransfDir, qsTmp);

		QString qsTransfFilename = qsTransfDir + "/transf" + padZeros(QString::number(imgidx + 1), 4) + ".mat";

		cout << "qsTransfFilename: " << qsTransfFilename.toStdString() << endl;
		assert(filesys::check_file(qsTransfFilename));

		FloatGrid2 _T13 = matlab_io::mat_load_multi_array<FloatGrid2>(qsTransfFilename, "T13");

		boost_math::double_matrix T13(3, 3);
		multi_array_op::array_to_matrix(_T13, T13);

		/** visualize sample in the original image */

		int nParts = part_app.m_part_conf.part_size();

		QImage img_original;
		assert(img_original.load(qsOriginalImg));
		QPainter painter(&img_original);
		painter.setRenderHints(QPainter::Antialiasing);
		painter.setPen(Qt::yellow);

		for (int pidx = 0; pidx < nParts; ++pidx) {

			int max_ridx = (int)all_samples[sampidx][pidx][0];
			int max_ix = (int)all_samples[sampidx][pidx][2];
			int max_iy = (int)all_samples[sampidx][pidx][1];  

			cout << "pidx: " << pidx << 
				", max_ridx: " << max_ridx << 
				", max_ix: " << max_ix << 
				", max_iy: " << max_iy << endl;

			int scaleidx = 0;

			PartBBox bbox;
			bbox_from_pos(part_app.m_exp_param, part_app.m_window_param.part(pidx), 
				scaleidx, max_ridx, max_ix, max_iy, 
				bbox);

			/* map bbox to original image */
			//       double_vector t(2);
			//       hc::map_point2(T13, bbox.part_pos, t);
			//       bbox.part_pos = t;

			hc::map_point2(T13, bbox.part_pos, bbox.part_pos);
			hc::map_vector(T13, bbox.part_x_axis, bbox.part_x_axis);
			hc::map_vector(T13, bbox.part_y_axis, bbox.part_y_axis);

			bbox.part_x_axis /= norm_2(bbox.part_x_axis);
			bbox.part_y_axis /= norm_2(bbox.part_y_axis);

			double scale = abs(T13(0, 0));

			bbox.min_proj_x *= scale;
			bbox.max_proj_x *= scale;

			bbox.min_proj_y *= scale;
			bbox.max_proj_y *= scale;

			draw_bbox(painter, bbox, 1, 2);

		}// parts 

		QString qsOutImg = qsBackprojectDir + "/imgidx_" + padZeros(QString::number(imgidx), 4) + ".png";
		cout << "saving " << qsOutImg.toStdString() << endl;

		assert(img_original.save(qsOutImg));
	}// iamges
}
