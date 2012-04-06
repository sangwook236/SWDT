//#include "stdafx.h"
#include <mrpt/core.h>
#include <deque>
#include <iostream>
#include <fstream>


using mrpt::utils::DEG2RAD;
using mrpt::utils::RAD2DEG;

namespace {
namespace local {

struct KfSlamOptions
{
	std::string configFileName;
	std::string rawSensorFileName;
	std::string rawOdometryFileName;

	std::string rawLogFileName;
	unsigned int rawLogOffset;
	std::string logOutputDirectoryName;
	int logFrequency;
	bool savePoseLog;
	bool saveMapRepresentation;

	std::string groundTruthFileName;

	bool save3DScene;
	bool camera3DSceneFollowsRobot;

	bool showProgress3DRealTime;
	int showProgress3DRealTimeDelayInMillisec;

	float insertionLinearDistance;
	float insertionAngularDistance;

	//mrpt::slam::TSetOfMetricMapInitializers metricMapsOptions;
	//mrpt::slam::CICP::TConfigParams icpOptions;

	//bool matchAgainstTheGrid;
};

void kf_slam_map_building(const KfSlamOptions &options, const bool useRawLogFile)
{
	mrpt::utils::CFileGZInputStream rawlogFile;
	std::ifstream streamSensor;
	std::ifstream streamOdometry;
	if (useRawLogFile)
	{
		if (!rawlogFile.open(options.rawLogFileName))
		{
			std::cerr << "raw log file open error !!!" << std::endl;
			return;
		}
	}
	else
	{
		streamSensor.open(options.rawSensorFileName.c_str());
		if (!streamSensor.is_open())
		{
			std::cerr << "raw sensor file open error !!!" << std::endl;
			return;
		}
		streamOdometry.open(options.rawOdometryFileName.c_str());
		if (!streamOdometry.is_open())
		{
			std::cerr << "raw odometry file open error !!!" << std::endl;
			return;
		}
	}

	// prepare output directory:
	mrpt::system::deleteFilesInDirectory(options.logOutputDirectoryName);
	mrpt::system::createDirectory(options.logOutputDirectoryName);

	// load the config options for mapping:
	mrpt::slam::CRangeBearingKFSLAM mapping;
	mapping.loadOptions(mrpt::utils::CConfigFile(options.configFileName));
	mapping.KF_options.dumpToConsole();
	mapping.options.dumpToConsole();

	// initialize map
	//mapping.initializeEmptyMap();

	// the main loop:
	mrpt::slam::CActionCollectionPtr actions;
	mrpt::slam::CSensoryFramePtr observations;

	std::size_t step = 0;
	std::deque<mrpt::slam::CPose3D> meanPath;  // the estimated path

	mrpt::slam::CPose3DPDFGaussian robotPose;
	std::vector<mrpt::slam::CPoint3D> landmarks;
	std::map<unsigned int, mrpt::slam::CLandmark::TLandmarkID> landmarkIds;
	mrpt::math::CMatrixDouble fullCov;
	mrpt::math::CVectorDouble fullState;

	std::size_t rawlogEntry = 0;
	for (;;)
	{
		if (mrpt::system::os::kbhit())
		{
			const char pushKey = mrpt::system::os::getch();
			//if (27 == pushKey)
				break;
		}

		if (useRawLogFile)
		{
			// load action/observation pair from the rawlog:
			if (!mrpt::slam::CRawlog::readActionObservationPair(rawlogFile, actions, observations, rawlogEntry))
				break;  // file EOF
		}
		else
		{
			if (streamSensor.eof() || streamOdometry.eof())
				break;

			// action information
			std::string odometryStr;
			std::getline(streamOdometry, odometryStr);

			mrpt::slam::CActionRobotMovement2D::TMotionModelOptions robotOdometryModelOptions;
			robotOdometryModelOptions.modelSelection = mrpt::slam::CActionRobotMovement2D::mmGaussian;
			robotOdometryModelOptions.gausianModel.a1 = 0.01f;  // ratio of motion to x/y std.dev [meters/meter]
			robotOdometryModelOptions.gausianModel.a2 = 0.001f * (180.0f / M_PIf);  // ratio of motion to phi std.dev [meters/degree]
			robotOdometryModelOptions.gausianModel.a3 = mrpt::utils::DEG2RAD(1.0f);  // ratio of rotation to x/y std.dev [degrees/meter]
			robotOdometryModelOptions.gausianModel.a4 = 0.05f;  // ratio of rotation to phi std.dev [degrees/degree]
			robotOdometryModelOptions.gausianModel.minStdXY = 0.01f;  // minimum std.dev of x/y [meters]
			robotOdometryModelOptions.gausianModel.minStdPHI = mrpt::utils::DEG2RAD(0.2f);  // minimum std.dev of phi [degrees]

			mrpt::slam::CActionRobotMovement2D robotOdometry;
			//robotOdometry.timestamp = mrpt::system::getCurrentTime();
			robotOdometry.estimationMethod = mrpt::slam::CActionRobotMovement2D::emOdometry;
			robotOdometry.hasEncodersInfo = false;
			robotOdometry.hasVelocities = false;
			//robotOdometry.motionModelConfiguration = robotOdometryModelOptions;

			std::istringstream sstreamOdometry(odometryStr);
			float x, y, phi;
			sstreamOdometry >> x >> y >> phi;
			mrpt::poses::CPose2D incOdometry;
			incOdometry.x(x);
			incOdometry.y(y);
			incOdometry.phi(phi);
			robotOdometry.computeFromOdometry(incOdometry, robotOdometryModelOptions);

			actions = mrpt::slam::CActionCollectionPtr(new mrpt::slam::CActionCollection());
			actions->insert(robotOdometry);

			// observation information
			const size_t sensorDataCount = 180;

			std::string sensorStr;
			std::getline(streamSensor, sensorStr);

			mrpt::slam::CObservation2DRangeScanPtr robotObservation(new mrpt::slam::CObservation2DRangeScan());
			//robotObservation->timestamp = mrpt::system::getCurrentTime();
			robotObservation->scan.reserve(sensorDataCount);
			robotObservation->validRange.reserve(sensorDataCount);
			robotObservation->aperture = M_PIf;
			robotObservation->rightToLeft = true;
			robotObservation->maxRange = 30.0f;
			//robotObservation->sensorPose = ;
			//robotObservation->stdError = ;
			//robotObservation->beamAperture = ;
			//robotObservation->deltaPitch = ;

			std::istringstream sstreamSensor(sensorStr);
			for (size_t i = 0; i < sensorDataCount; ++i)
			{
				float range;
				sstreamSensor >> range;
				robotObservation->scan.push_back(range);
				robotObservation->validRange.push_back(range <= robotObservation->maxRange);
			}

			observations = mrpt::slam::CSensoryFramePtr(new mrpt::slam::CSensoryFrame());;
			observations->insert(robotObservation);

			++rawlogEntry;
		}

		if (rawlogEntry >= options.rawLogOffset)
		{
			// process the action and observations:
			mapping.processActionObservation(actions, observations);

			// get current state:
			mapping.getCurrentState(robotPose, landmarks, landmarkIds, fullState, fullCov);
			std::cout << "mean pose: " << robotPose.mean << std::endl;
			std::cout << "# of landmarks in the map: " << landmarks.size() << std::endl;

			// build the path:
			meanPath.push_back(robotPose.mean);

			// save mean pose:
			if (!(step % options.logFrequency))
			{
				const mrpt::slam::CPose3D robPose3D(robotPose.mean);
				const mrpt::math::CMatrix robotPose((mrpt::math::TPose3D)robPose3D);
				robotPose.saveToTextFile(options.logOutputDirectoryName + mrpt::utils::format("/robot_pose_%05u.txt", (unsigned int)step));
			}

			// save full cov:
			if (!(step % options.logFrequency))
			{
				fullCov.saveToTextFile(options.logOutputDirectoryName + mrpt::utils::format("/full_cov_%05u.txt", (unsigned int)step));
			}

			// save map to file representations?
			if (options.saveMapRepresentation  && !(step % options.logFrequency))
			{
				mapping.saveMapAndPath2DRepresentationAsMATLABFile(options.logOutputDirectoryName + mrpt::utils::format("/slam_state_%05u.m", (unsigned int)step));
			}

			// save 3D view of the filter state:
			if (options.save3DScene && !(step % options.logFrequency))
			{
				mrpt::opengl::COpenGLScene scene3D;
				{
					mrpt::opengl::CGridPlaneXYPtr grid = mrpt::opengl::CGridPlaneXY::Create(-1000, 1000, -1000, 1000, 0, 5);
					grid->setColor(0.4, 0.4, 0.4);
					scene3D.insert(grid);
				}

				// robot path:
				{
					mrpt::opengl::CSetOfLinesPtr linesPath = mrpt::opengl::CSetOfLines::Create();
					linesPath->setColor(1,0,0);

					double x0 = 0, y0 = 0, z0 = 0;

					if (!meanPath.empty())
					{
						x0 = meanPath[0].x();
						y0 = meanPath[0].y();
						z0 = meanPath[0].z();
					}

					for (std::deque<mrpt::slam::CPose3D>::iterator it = meanPath.begin(); it != meanPath.end(); ++it)
					{
						linesPath->appendLine(
							x0, y0, z0,
							it->x(), it->y(), it->z()
						);
						x0 = it->x();
						y0 = it->y();
						z0 = it->z();
					}
					scene3D.insert(linesPath);
				}

				{
					mrpt::opengl::CSetOfObjectsPtr objs = mrpt::opengl::CSetOfObjects::Create();
					mapping.getAs3DObject(objs);
					scene3D.insert(objs);
				}

				// save to file:
				mrpt::utils::CFileGZOutputStream(options.logOutputDirectoryName + mrpt::utils::format("/kf_state_%05u.3Dscene", (unsigned int)step)) << scene3D;
			}
		}

		std::cout << mrpt::utils::format("\nStep %u  - Rawlog entries processed: %i\n", (unsigned int)step, (unsigned int)rawlogEntry);

		// free rawlog items memory:
		actions.clear_unique();
		observations.clear_unique();

		++step;
	};	// end "while(1)"


	// compute the "information" between partitions:
	if (mapping.options.doPartitioningExperiment)
	{
		// PART I: comparison to fixed partitioning every K obs.

		// compute the information matrix:
		size_t i;
		for (i = 0; i < 6; ++i) fullCov(i,i) = std::max(fullCov(i,i), 1e-6);

		mrpt::math::CMatrix H(fullCov.inv());
		H.saveToTextFile(options.logOutputDirectoryName + std::string("/information_matrix_final.txt"));

		// replace by absolute values:
		H.Abs();
		mrpt::math::CMatrix H2(H);
		//--S [] 2012/04/06: Sang-Wook Lee
		// TODO [check] >> are it changed correctly?
		//H2.adjustRange();
		H2.adjustRange(0.0f, 1.0f);
		//--E 2012/04/06
		mrpt::utils::CMRPTImageFloat imgF(H2);
		imgF.saveToFile(options.logOutputDirectoryName + std::string("/information_matrix_final.png"));


		// compute the "approximation error factor" E:
		//  E = SUM() / SUM(ALL ELEMENTS IN MATRIX)
		std::vector<mrpt::vector_uint> landmarksMembership, partsInObsSpace;
		mrpt::math::CMatrix ERRS(50,3);

		for (i = 0; i < ERRS.getRowCount(); ++i)
		{
			size_t K;

			if (0 == i)
			{
				K=0;
				mapping.getLastPartitionLandmarks(landmarksMembership);
			}
			else
			{
				K = i + 1;
				mapping.getLastPartitionLandmarksAsIfFixedSubmaps(i + 1, landmarksMembership);
			}

			mapping.getLastPartition(partsInObsSpace);

			ERRS(i,0) = (float)K;
			ERRS(i,1) = (float)partsInObsSpace.size();
			ERRS(i,2) = mapping.computeOffDiagonalBlocksApproximationError(landmarksMembership);
		}

		ERRS.saveToTextFile(options.logOutputDirectoryName + std::string("/ERRORS.txt"));
		//printf("Approximation error from partition:\n"); std::cout << ERRS << std::endl;

		// PART II: sweep partitioning threshold:
		size_t STEPS = 50;
		mrpt::math::CVectorFloat ERRS_SWEEP(STEPS), ERRS_SWEEP_THRESHOLD(STEPS);

		// compute the error for each partitioning-threshold
		for (i = 0; i < STEPS; ++i)
		{
			const float th = (1.0f * i) / (STEPS - 1.0f);
			ERRS_SWEEP_THRESHOLD[i] = th;
			mapping.mapPartitionOptions()->partitionThreshold = th;

			mapping.reconsiderPartitionsNow();

			mapping.getLastPartitionLandmarks(landmarksMembership);
			ERRS_SWEEP[i] = mapping.computeOffDiagonalBlocksApproximationError(landmarksMembership);
		}

		ERRS_SWEEP.saveToTextFile(options.logOutputDirectoryName + std::string("/ERRORS_SWEEP.txt"));
		ERRS_SWEEP_THRESHOLD.saveToTextFile(options.logOutputDirectoryName + std::string("/ERRORS_SWEEP_THRESHOLD.txt"));

	}  // end if doPartitioningExperiment


	// is there ground truth??
	if (!options.groundTruthFileName.empty() && mrpt::utils::fileExists(options.groundTruthFileName))
	{
		mrpt::math::CMatrixFloat GT(0, 0);
		try
		{
			GT.loadFromTextFile(options.groundTruthFileName);
		}
		catch (const std::exception &e)
		{
			std::cerr << "ignoring the following error loading ground truth file: " << e.what() << std::endl;
		}

		if (GT.getRowCount() > 0 && !landmarks.empty())
		{
			// each row has:
			//   [0] [1] [2]  [6]
			//    x   y   z    ID
			mrpt::vector_double ERRS(0);
			for (size_t i = 0; i < landmarks.size(); ++i)
			{
				// Find the entry in the GT for this mapped LM:
				bool found = false;
				for (size_t r = 0; r < GT.getRowCount(); ++r)
				{
					if (landmarkIds[i] == GT(r,6))
					{
						ERRS.push_back(landmarks[i].distance3DTo(GT(r,0), GT(r,1), GT(r,2)));
						found = true;
						break;
					}
				}
				if (!found)
				{
					std::cerr << "ground truth entry not found for landmark ID:" << landmarkIds[i] << std::endl;
				}
			}

			std::cout << "ERRORS VS. GROUND TRUTH:" << std::endl;
			std::cout << "mean error: " << mrpt::math::mean(ERRS) << " meters" << std::endl;
			std::cout << "minimum error: " << mrpt::math::minimum(ERRS) << " meters" << std::endl;
			std::cout << "maximum error: " << mrpt::math::maximum(ERRS) << " meters" << std::endl;
		}

	} // end if GT
}

}  // namespace local
}  // unnamed namespace

void slam_kf()
{
	const bool useRawLogFile = true;
	const std::string rawSensorFileName("mrpt_data\\slam_kf\\kf-slam_demo.rawlog_LASER.txt");
	const std::string rawOdometryFileName("mrpt_data\\slam_kf\\kf-slam_demo.rawlog_ODO.txt");

	const std::string INI_FILENAME("mrpt_data\\slam_kf\\config_slam.ini");
	if (!mrpt::system::fileExists(INI_FILENAME))
	{
		std::cout << "ini file not found !!!" << std::endl;
		return;
	}

	mrpt::utils::CConfigFile iniFile(INI_FILENAME);

	local::KfSlamOptions options;

	// load config from file
	options.configFileName = INI_FILENAME;
	//--S [] 2009/08/08: Sang-Wook Lee
	options.rawSensorFileName = rawSensorFileName;
	options.rawOdometryFileName = rawOdometryFileName;
	//--E [] 2009/08/08
	options.rawLogFileName = iniFile.read_string("MappingApplication", "rawlog_file", "", /*force existence:*/ true);
	options.rawLogOffset = iniFile.read_int("MappingApplication", "rawlog_offset", 0, /*force existence:*/ false);
	options.logOutputDirectoryName = iniFile.read_string("MappingApplication", "logOutput_dir", "log_out", /*force existence:*/ false);
	options.logFrequency = iniFile.read_int("MappingApplication", "LOG_FREQUENCY", 5, /*force existence:*/ false);
	options.savePoseLog = iniFile.read_bool("MappingApplication", "SAVE_POSE_LOG", false, /*force existence:*/ false);
	options.saveMapRepresentation = iniFile.read_bool("MappingApplication", "SAVE_MAP_REPRESENTATION", /*force existence:*/ true);

	options.groundTruthFileName = iniFile.read_string("MappingApplication", "ground_truth_file", "", /*force existence:*/ false);

	options.save3DScene = iniFile.read_bool("MappingApplication", "SAVE_3D_SCENE", false, /*force existence:*/ true);
	options.camera3DSceneFollowsRobot = iniFile.read_bool("MappingApplication", "CAMERA_3DSCENE_FOLLOWS_ROBOT", true, /*force existence:*/ false);

	bool SHOW_PROGRESS_3D_REAL_TIME = false;
	int SHOW_PROGRESS_3D_REAL_TIME_DELAY_MS = 0;
	MRPT_LOAD_CONFIG_VAR(SHOW_PROGRESS_3D_REAL_TIME, bool, iniFile, "MappingApplication");
	MRPT_LOAD_CONFIG_VAR(SHOW_PROGRESS_3D_REAL_TIME_DELAY_MS, int, iniFile, "MappingApplication");
	options.showProgress3DRealTime = SHOW_PROGRESS_3D_REAL_TIME;
	options.showProgress3DRealTimeDelayInMillisec = SHOW_PROGRESS_3D_REAL_TIME_DELAY_MS;

	float insertionLinDistance = 0.0f;
	float insertionAngDistance = 0.0f;
	MRPT_LOAD_CONFIG_VAR(insertionLinDistance, float, iniFile, "MappingApplication");
	MRPT_LOAD_CONFIG_VAR_DEGREES(insertionAngDistance, iniFile, "MappingApplication");
	options.insertionLinearDistance = insertionLinDistance;
	options.insertionAngularDistance = insertionAngDistance;

	//options.metricMapsOptions.loadFromConfigFile(iniFile, "MappingApplication");
	//options.icpOptions.loadFromConfigFile(iniFile, "ICP");
	//options.matchAgainstTheGrid = iniFile.read_bool("MappingApplication", "matchAgainstTheGrid", true);

	// print params:
	std::cout << "running with the following parameters:" << std::endl;
	std::cout << " RAWLOG file:'" << options.rawLogFileName << "'" << std::endl;
	std::cout << " output directory:\t\t'" << options.logOutputDirectoryName << "'" << std::endl;
	//std::cout << " matchAgainstTheGrid:\t\t\t" << (options.matchAgainstTheGrid ? 'Y' : 'N') << std::endl;
	std::cout << " log record freq:\t\t" << options.logFrequency << std::endl;
	std::cout << "  savePoseLog:\t\t\t" << (options.savePoseLog ? 'Y' : 'N') << std::endl;
	std::cout << "  saveMapRepresentation:\t" << (options.saveMapRepresentation ? 'Y' : 'N') << std::endl;
	std::cout << "  save3DScene:\t\t\t" << (options.save3DScene ? 'Y' : 'N') << std::endl;
	std::cout << "  camera3DSceneFollowsRobot:\t" << (options.camera3DSceneFollowsRobot ? 'Y' : 'N') << std::endl;

	std::cout << std::endl;

	// output options to console
	//options.metricMapsOptions.dumpToConsole();
	//options.icpOptions.dumpToConsole();

	if (options.rawLogFileName.empty() || !mrpt::system::fileExists(options.rawLogFileName))
	{
		std::cout << "raw log file error !!!" << std::endl;
		return;
	}

	std::cout << "\nmap building ...." << std::endl;
	local::kf_slam_map_building(options, useRawLogFile);
}
