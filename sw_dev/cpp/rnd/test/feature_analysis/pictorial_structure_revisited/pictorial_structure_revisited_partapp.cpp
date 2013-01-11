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

#include <QFile>
#include <QTextStream>
#include <QString>

#include <boost/program_options.hpp>

#include <libPartApp/partapp.h>
#include <libPartApp/partapp_aux.hpp>

#include <libPartDetect/partdetect.h>

#include <libMisc/misc.hpp>

#include <libFilesystemAux/filesystem_aux.h>

#include <libPictStruct/objectdetect.h>

// this is only needed for szOption_EvalGt
#include <libPartDetect/partdetect.h>

#include <libMatlabIO/matlab_io.h>
#include <libMatlabIO/matlab_io.hpp>

#include "parteval.h"

using namespace std;
namespace po = boost::program_options;

namespace {
namespace local {

const char* szOption_Help = "help";
const char* szOption_ExpOpt = "expopt";
const char* szOption_TrainClass = "train_class";
const char* szOption_TrainBootstrap = "train_bootstrap";
const char* szOption_Pidx = "pidx";

const char* szOption_BootstrapPrep = "bootstrap_prep";
const char* szOption_BootstrapDetect = "bootstrap_detect";
const char* szOption_BootstrapShowRects = "bootstrap_showrects";

const char* szOption_First = "first";
const char* szOption_NumImgs = "numimgs";

const char* szOption_PartDetect = "part_detect";
const char* szOption_FindObj = "find_obj";
const char* szOption_pc_learn = "pc_learn";
const char* szOption_SaveRes = "save_res";

const char* szOption_EvalSegments = "eval_segments";

const char* szOption_Distribute = "distribute";
const char* szOption_NCPU = "ncpu";
const char* szOption_BatchNumber = "batch_num";

const char* szOption_VisParts = "vis_parts";

/**
initialize first and last indices from command line parameters 

check that indices are in the valid range

(this could also be used to support automatic splitting of data in chunks, see old code)
*/

void init_firstidx_lastidx(const AnnotationList &annolist, po::variables_map cmd_vars_map, int &firstidx, int &lastidx)
{
	int param_firstidx;
	int param_lastidx;

	if (cmd_vars_map.count(szOption_First))
		param_firstidx = cmd_vars_map[szOption_First].as<int>();
	else
		param_firstidx = 0;

	if (cmd_vars_map.count(szOption_NumImgs)) { 
		param_lastidx = param_firstidx + cmd_vars_map[szOption_NumImgs].as<int>() - 1;
	}
	else {
		param_lastidx = annolist.size() - 1;
	}

	if (cmd_vars_map.count(szOption_Distribute)) {
		assert(cmd_vars_map.count(szOption_NCPU));
		assert(cmd_vars_map.count(szOption_BatchNumber));

		int ncpu = cmd_vars_map[szOption_NCPU].as<int>();
		int batch_num = cmd_vars_map[szOption_BatchNumber].as<int>();

		int num_per_cpu = (int)ceil((param_lastidx - param_firstidx + 1)/(float)ncpu);
		firstidx = param_firstidx + batch_num*num_per_cpu;
		lastidx = firstidx + num_per_cpu - 1;
		cout << "ncpu: " << ncpu << ", num_per_cpu: " << num_per_cpu << endl;
	}
	else {
		firstidx = param_firstidx;
		lastidx = param_lastidx;
	}

	/* allow firstidx to be > then lastidx, no images will be processed in this case */
	check_bounds_and_update(firstidx, 0, (int)annolist.size() + 1);
	check_bounds_and_update(lastidx, 0, (int)annolist.size());  
}

}  // namespace local
}  // unnamed namespace

namespace my_pictorial_structure_revisited {

int pictorial_structure_revisited_partapp_main(int argc, char *argv[])
{
	/* parse command line options */
	po::options_description cmd_options_desc("command line options:");
	po::variables_map cmd_vars_map;

	cmd_options_desc.add_options()
		(local::szOption_Help, "help message")
		(local::szOption_ExpOpt, po::value<string>(), "experiment parameters")   
		(local::szOption_TrainClass, "train part detector")
		(local::szOption_TrainBootstrap, "train part detector, adding hard negative samples")
		(local::szOption_Pidx, po::value<int>(), "0-based index of the part")

		(local::szOption_BootstrapPrep, "create bootstrapping dataset (crops of objects with some background)")
		(local::szOption_BootstrapDetect, "run previously trained classifer on bootstrapping images")
		(local::szOption_BootstrapShowRects, "show top negatives on bootstrapping images")

		(local::szOption_First, po::value<int>(), "index of first image")
		(local::szOption_NumImgs, po::value<int>(), "number of images to process")

		(local::szOption_PartDetect, "run part detector on the test set")

		(local::szOption_FindObj, "find_obj")
		(local::szOption_pc_learn, "estimate prior on part configurations with maximum likelihood")
		(local::szOption_SaveRes, "save object recognition results in al/idl formats")

		(local::szOption_EvalSegments, "evaluate part localization according to Ferrari's criteria")

		(local::szOption_Distribute, "split processing into multiple chunks, keep track of the chunks which must be processed")
		(local::szOption_NCPU, po::value<int>(), "number of chunks")
		(local::szOption_BatchNumber, po::value<int>(), "current chunk")
		(local::szOption_VisParts, "visualize ground-truth part positions")
		;


	/* BEGIN: handle 'distribute' option */

	const QString qsArgsFile = ".curexp";
	const QString qsCurBatchFile = ".curexp_batchnum";
	const int MAX_ARGC = 255;
	const int MAX_ARG_LENGTH = 1024;
	cout << "argc: " << argc << endl;
	cout << "argv[0]: " << argv[0] << endl;
	assert(argc <= MAX_ARGC);

	int loaded_argc = 0;
	char loaded_argv[MAX_ARGC][MAX_ARG_LENGTH];
	char *loaded_argv_ptr[MAX_ARGC];
	for (int idx = 0; idx < MAX_ARGC; ++idx) {
		loaded_argv_ptr[idx] = loaded_argv[idx];
	}

	/* try to load parameters from experiment file if none are specified on the command line */
	if (argc <= 1 && filesys::check_file(qsArgsFile)) {
		// load .curexp, fill argc and argv
		cout << "reading arguments from " << qsArgsFile.toStdString() << endl;
		QFile qfile( qsArgsFile );

		assert(qfile.open( QIODevice::ReadOnly | QIODevice::Text));
		QTextStream stream( &qfile );
		QString line;

		while(!(line = stream.readLine()).isNull()) {
			if (!line.isEmpty()) {
				cout << line.toStdString() << endl;
				assert(line.length() < MAX_ARG_LENGTH);

				int cur_idx = loaded_argc;
				strcpy(loaded_argv[cur_idx], line.toStdString().c_str());
				++loaded_argc;
			}
		}

	}

	/* process command option */

	try {

		if (loaded_argc > 0) {
			cout << "reading parameters from " << qsArgsFile.toStdString() << endl;
			po::store(po::parse_command_line(loaded_argc, loaded_argv_ptr, cmd_options_desc), cmd_vars_map);

			ifstream ifs;
			ifs.open(qsCurBatchFile.toStdString().c_str());
			po::store(po::parse_config_file(ifs, cmd_options_desc), cmd_vars_map);
			ifs.close();

			assert(cmd_vars_map.count("batch_num") > 0);
			int batch_num = cmd_vars_map["batch_num"].as<int>();
			cout << "batch_num: " << batch_num << endl;
			++batch_num;

			cout << "updating " << qsCurBatchFile.toStdString() << endl;
			QFile qfile2(qsCurBatchFile);
			assert(qfile2.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text));
			QTextStream stream2( &qfile2 );
			stream2 << "batch_num = " << batch_num << endl;
		}
		else {
			cout << "reading command line parameters" << endl;
			po::store(po::parse_command_line(argc, argv, cmd_options_desc), cmd_vars_map);
		}

		po::notify(cmd_vars_map);
	}
	catch (exception &e) {
		cerr << "error: " << e.what() << endl;
		return 1;
	}  

	/* "distribute" option means store parameters and initialize batch counter */
	if (argc > 1 && cmd_vars_map.count("distribute") > 0) {
		assert(cmd_vars_map.count("ncpu") > 0);

		cout << "saving command line parameters in " << qsArgsFile.toStdString() << endl;
		QFile qfile( qsArgsFile );
		assert(qfile.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text));
		QTextStream stream( &qfile );

		for (int idx = 0; idx < argc; ++idx) 
			stream << argv[idx] << endl;

		/* how to initialize program option explicitly ??? */
		cout << "intializing batch counter: " << qsCurBatchFile.toStdString() << endl;
		{
			QFile qfile2(qsCurBatchFile);
			assert(qfile2.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text));
			{
				QTextStream stream2( &qfile2 );
				stream2 << "batch_num = 0" << endl;
			}
			qfile2.close();
		}

		ifstream ifs;
		ifs.open(qsCurBatchFile.toStdString().c_str());
		po::store(po::parse_config_file(ifs, cmd_options_desc), cmd_vars_map);
		ifs.close();

		{
			QFile qfile2(qsCurBatchFile);
			assert(qfile2.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text));
			{
				QTextStream stream2( &qfile2 );
				stream2 << "batch_num = 1" << endl;
			}
			qfile2.close();
		}
	}

	/* END: handle 'distribute' option */


	/* process options which do not require initialized partapp */
	if (cmd_vars_map.count(local::szOption_Help)) {
		cout << cmd_options_desc << endl << endl;
		return 1;
	}

	/* initialize partapp from parameter file */
	if (cmd_vars_map.count(local::szOption_ExpOpt) == 0) {
		cout << cmd_options_desc << endl << endl;
		cout << "'expopt' parameter missing" << endl;
		return 1;
	}

	PartApp part_app;
	QString qsExpParam = cmd_vars_map[local::szOption_ExpOpt].as<string>().c_str();
	cout << "initializing from " << qsExpParam.toStdString() << endl;
	part_app.init(qsExpParam);  

	if (cmd_vars_map.count(local::szOption_TrainClass)) {
		bool bBootstrap = cmd_vars_map.count(local::szOption_TrainBootstrap); // default is false

		if (cmd_vars_map.count(local::szOption_Pidx)) {

			int pidx = cmd_vars_map[local::szOption_Pidx].as<int>();
			assert(pidx < part_app.m_part_conf.part_size());
			assert(part_app.m_part_conf.part(pidx).is_detect());

			cout << "training classifier for part " << pidx << endl;
			part_detect::abc_train_class(part_app, pidx, bBootstrap);
		}
		else {
			for (int pidx = 0; pidx < part_app.m_part_conf.part_size(); ++pidx) {
				if (part_app.m_part_conf.part(pidx).is_detect()) {
					cout << "training classifier for part " << pidx << endl;
					part_detect::abc_train_class(part_app, pidx, bBootstrap);
				}
			}
		}

	}// train class
	else if (cmd_vars_map.count(local::szOption_BootstrapPrep)) {
		part_detect::prepare_bootstrap_dataset(part_app, part_app.m_train_annolist, 0, part_app.m_train_annolist.size() - 1);
	}
	else if (cmd_vars_map.count(local::szOption_BootstrapDetect)) {
		int firstidx, lastidx;
		local::init_firstidx_lastidx(part_app.m_train_annolist, cmd_vars_map, firstidx, lastidx);

		part_detect::bootstrap_partdetect(part_app, firstidx, lastidx);
	}
	else if (cmd_vars_map.count(local::szOption_BootstrapShowRects)) {
		assert(cmd_vars_map.count("pidx") && "part index missing");
		int pidx = cmd_vars_map["pidx"].as<int>();

		int firstidx, lastidx;
		local::init_firstidx_lastidx(part_app.m_train_annolist, cmd_vars_map, firstidx, lastidx);

		int num_rects = 50;
		double min_score = 0.1;
		vector<PartBBox> v_rects;
		vector<double> v_rects_scale;

		bool bIgnorePartRects = true;
		bool bDrawRects = true;

		/* create debug directory to save bootstrap images */
		if (!filesys::check_dir("./debug")) {
			cout << "creating ./debug" << endl;
			filesys::create_dir("./debug");
		}

		part_detect::bootstrap_get_rects(part_app, firstidx, pidx, num_rects, min_score, 
			v_rects, v_rects_scale,
			bIgnorePartRects, bDrawRects);

	}
	else if (cmd_vars_map.count(local::szOption_pc_learn)) {
		object_detect::learn_conf_param(part_app, part_app.m_train_annolist);
	}
	else if (cmd_vars_map.count(local::szOption_SaveRes)) {
		int scoreProbMapType = object_detect::SPMT_NONE;

		object_detect::saveRecoResults(part_app, scoreProbMapType);
	}
	else if (cmd_vars_map.count(local::szOption_EvalSegments)) {
		int firstidx, lastidx;
		local::init_firstidx_lastidx(part_app.m_test_annolist, cmd_vars_map, firstidx, lastidx);
		cout << "processing images: " << firstidx << " to " << lastidx << endl;

		eval_segments(part_app, firstidx, lastidx);

	}
	else if (cmd_vars_map.count(local::szOption_VisParts)) {

		int firstidx, lastidx;
		local::init_firstidx_lastidx(part_app.m_train_annolist, cmd_vars_map, firstidx, lastidx);

		for (int imgidx = firstidx; imgidx <= lastidx; ++imgidx) {
			QImage img = visualize_parts(part_app.m_part_conf, part_app.m_window_param, part_app.m_train_annolist[imgidx]);

			QString qsPartConfPath;
			QString qsPartConfName;
			QString qsPartConfExt;
			filesys::split_filename_ext(part_app.m_exp_param.part_conf().c_str(), qsPartConfPath, qsPartConfName, qsPartConfExt);

			QString qsDebugDir = (part_app.m_exp_param.log_dir() + "/" + 
				part_app.m_exp_param.log_subdir() + "/debug").c_str();

			if (!filesys::check_dir(qsDebugDir))
				filesys::create_dir(qsDebugDir);

			QString qsFilename = qsDebugDir + "/parts-" + qsPartConfName + "-imgidx" + 
				padZeros(QString::number(imgidx), 4) + ".png";
			cout << "saving " << qsFilename.toStdString() << endl;

			assert(img.save(qsFilename));    
		}

	}
	else {
		bool bShowHelpMessage = true;

		if (cmd_vars_map.count(local::szOption_PartDetect)) {

			int firstidx, lastidx;
			local::init_firstidx_lastidx(part_app.m_test_annolist, cmd_vars_map, firstidx, lastidx);

			bool bSaveImageScoreGrid = part_app.m_exp_param.save_image_scoregrid();
			cout << "bSaveImageScoreGrid: " << bSaveImageScoreGrid << endl;

			part_detect::partdetect(part_app, firstidx, lastidx, false, bSaveImageScoreGrid);
			if (part_app.m_exp_param.flip_orientation())
				part_detect::partdetect(part_app, firstidx, lastidx, true, bSaveImageScoreGrid);

			bShowHelpMessage = false;
		}

		if (cmd_vars_map.count(local::szOption_FindObj)) {
			int firstidx, lastidx;
			local::init_firstidx_lastidx(part_app.m_test_annolist, cmd_vars_map, firstidx, lastidx);

			int scoreProbMapType = object_detect::SPMT_NONE;
			object_detect::findObjectDataset(part_app, firstidx, lastidx, scoreProbMapType);

			bShowHelpMessage = false;
		}

		if (bShowHelpMessage) {
			cout << cmd_options_desc << endl;
			return 1;
		}
	}
	
	return 0;
}

}  // namespace my_pictorial_structure_revisited
