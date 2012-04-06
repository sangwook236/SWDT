//#include "stdafx.h"
#include <mrpt/core.h>
#include <iostream>
#include <fstream>


using mrpt::utils::DEG2RAD;
using mrpt::utils::RAD2DEG;

namespace {
namespace local {

struct IcpSlamOptions
{
	std::string rawSensorFileName;
	std::string rawOdometryFileName;

	std::string rawLogFileName;
	unsigned int rawLogOffset;
	std::string logOutputDirectoryName;
	int logFrequency;
	bool savePoseLog;

	bool save3DScene;
	bool camera3DSceneFollowsRobot;

	bool showProgress3DRealTime;
	int showProgress3DRealTimeDelayInMillisec;

	float insertionLinearDistance;
	float insertionAngularDistance;

	mrpt::slam::TSetOfMetricMapInitializers metricMapsOptions;
	mrpt::slam::CICP::TConfigParams icpOptions;

	bool matchAgainstTheGrid;
};

void icp_slam_map_building(const IcpSlamOptions &options, const bool useRawLogFile)
{
	//MRPT_TRY_START

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

	//
	int step = 0;
	float t_exec;
	mrpt::utils::CTicTac tictac, tictacGlobal;
	mrpt::slam::CSensFrameProbSequence finalMap;
	mrpt::slam::COccupancyGridMap2D::TEntropyInfo entropy;

	// constructor
	mrpt::slam::CMetricMapBuilderICP mapBuilder(
		(mrpt::slam::TSetOfMetricMapInitializers *)&options.metricMapsOptions,
		options.insertionLinearDistance,
		options.insertionAngularDistance,
		(mrpt::slam::CICP::TConfigParams *)&options.icpOptions
	);

	mapBuilder.ICP_options.matchAgainstTheGrid = options.matchAgainstTheGrid;

	// set CMetricMapBuilder::TOptions
	mapBuilder.options.verbose = true;
	mapBuilder.options.enableMapUpdating = true;
	mapBuilder.options.debugForceInsertion = false;
	mapBuilder.options.insertImagesAlways = false;

	// prepare output directory:
	mrpt::system::deleteFilesInDirectory(options.logOutputDirectoryName);
	mrpt::system::createDirectory(options.logOutputDirectoryName);

	// open log files:
	mrpt::utils::CFileOutputStream f_log(mrpt::format("%s/log_times.txt", options.logOutputDirectoryName.c_str()));
	mrpt::utils::CFileOutputStream f_path(mrpt::format("%s/log_estimated_path.txt", options.logOutputDirectoryName.c_str()));
	mrpt::utils::CFileOutputStream f_pathOdo(mrpt::format("%s/log_odometry_path.txt", options.logOutputDirectoryName.c_str()));

	// create 3D window if requested:
	mrpt::gui::CDisplayWindow3DPtr win3D;
	if (options.showProgress3DRealTime)
	{
		win3D = mrpt::gui::CDisplayWindow3DPtr(new mrpt::gui::CDisplayWindow3D("ICP-SLAM @ MRPT C++ Library (C) 2004-2008", 600, 500));
		win3D->setCameraZoom(20);
		win3D->setCameraAzimuthDeg(-45);
	}

	// map building
	mrpt::slam::CActionCollectionPtr actions;
	mrpt::slam::CSensoryFramePtr observations;
	mrpt::poses::CPose2D odoPose(0, 0, 0);

	tictacGlobal.Tic();
	size_t rawlogEntry = 0;
	for (;;)
	{
		if (mrpt::system::os::kbhit())
		{
			const char c = mrpt::system::os::getch();
			//if (27 == c)
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
			// update odometry:
			{
				mrpt::slam::CActionRobotMovement2DPtr act = actions->getBestMovementEstimation();
				if (act)
					odoPose = odoPose + act->poseChange->getMeanVal();
			}

			// execute:
			tictac.Tic();
			mapBuilder.processActionObservation(*actions, *observations);
			t_exec = tictac.Tac();
			std::cout << "map building executed in " << (1000.0f * t_exec) << "ms" << std::endl;

			// info log:-
			f_log.printf("%f %d\n", 1000.0f * t_exec, mapBuilder.getCurrentlyBuiltMapSize());

			const mrpt::slam::CMultiMetricMap *mostLikMap = mapBuilder.getCurrentlyBuiltMetricMap();

			if (0 == (step % options.logFrequency))
			{
				// pose log:
				if (options.savePoseLog)
				{
					std::cout << "saving pose log information...";
					mapBuilder.getCurrentPoseEstimation()->saveToTextFile(mrpt::utils::format("%s/mapbuild_posepdf_%03u.txt", options.logOutputDirectoryName.c_str(), step));
					std::cout << "Ok" << std::endl;
				}
			}

			// save a 3D scene view of the mapping process:
			if (0 == (step % options.logFrequency) || (options.save3DScene || win3D.present()))
			{
				mrpt::poses::CPose3D robotPose;
				mapBuilder.getCurrentPoseEstimation()->getMean(robotPose);

				mrpt::opengl::COpenGLScenePtr scene = mrpt::opengl::COpenGLScene::Create();

				mrpt::opengl::COpenGLViewportPtr view = scene->getViewport("main");
				if (!view)
				{
					std::cout << "OpenGL viewport creation error !!!" << std::endl;
					return;
				}

				mrpt::opengl::COpenGLViewportPtr view_map = scene->createViewport("mini-map");
				view_map->setBorderSize(2);
				view_map->setViewportPosition(0.01, 0.01, 0.35, 0.35);
				view_map->setTransparent(false);

				{
					mrpt::opengl::CCamera &cam = view_map->getCamera();
					cam.setAzimuthDegrees(-90);
					cam.setElevationDegrees(90);
					cam.setPointingAt(robotPose);
					cam.setZoomDistance(20);
					cam.setOrthogonal();
				}

				// draw the ground floor
				mrpt::opengl::CGridPlaneXYPtr groundPlane = mrpt::opengl::CGridPlaneXY::Create(-200, 200, -200, 200, 0, 5);
				groundPlane->setColor(0.4, 0.4, 0.4);
				view->insert(groundPlane);
				view_map->insert(mrpt::opengl::CRenderizablePtr(groundPlane));  // a copy

				// set the camera pointing to the current robot pose:
				if (options.camera3DSceneFollowsRobot)
				{
					scene->enableFollowCamera(true);

					mrpt::opengl::CCamera &cam = view_map->getCamera();
					cam.setAzimuthDegrees(-45);
					cam.setElevationDegrees(45);
					cam.setPointingAt(robotPose);
				}

				// draw the maps
				{
					mrpt::opengl::CSetOfObjectsPtr obj = mrpt::opengl::CSetOfObjects::Create();
					mostLikMap->getAs3DObject(obj);
					view->insert(obj);

					// only the point map:
					mrpt::opengl::CSetOfObjectsPtr ptsMap = mrpt::opengl::CSetOfObjects::Create();
					//if (mostLikMap->m_pointsMaps.size())
					if (!mostLikMap->m_pointsMaps.empty())
					{
						mostLikMap->m_pointsMaps[0]->getAs3DObject(ptsMap);
						view_map->insert(ptsMap);
					}
				}

				// draw the robot path:
				mrpt::poses::CPose3DPDFPtr posePDF = mapBuilder.getCurrentPoseEstimation();
				mrpt::poses::CPose3D curRobotPose;
				posePDF->getMean(curRobotPose);
				{
					mrpt::opengl::CSetOfObjectsPtr obj = mrpt::opengl::stock_objects::RobotPioneer();
					obj->setPose(curRobotPose);
					view->insert(obj);
				}
				{
					mrpt::opengl::CSetOfObjectsPtr obj = mrpt::opengl::stock_objects::RobotPioneer();
					obj->setPose(curRobotPose);
					view_map->insert(obj);
				}

				// save as file:
				if (0 == (step % options.logFrequency) && options.save3DScene)
				{
					mrpt::utils::CFileGZOutputStream f(mrpt::utils::format("%s/buildingmap_%05u.3Dscene", options.logOutputDirectoryName.c_str(), step));
					f << *scene;
				}

				// show 3D?
				if (win3D)
				{
					mrpt::opengl::COpenGLScenePtr &ptrScene = win3D->get3DSceneAndLock();
					ptrScene = scene;

					win3D->unlockAccess3DScene();

					// move camera:
					win3D->setCameraPointingToPoint(curRobotPose.x(), curRobotPose.y(), curRobotPose.z());

					// update:
					win3D->forceRepaint();

					mrpt::system::sleep(options.showProgress3DRealTimeDelayInMillisec);
				}
			}

			// save the memory usage:
			{
				std::cout << "saving memory usage...";
				const unsigned long memUsage = mrpt::system::getMemoryUsage();
				FILE *f = mrpt::system::os::fopen(mrpt::utils::format("%s/log_MemoryUsage.txt", options.logOutputDirectoryName.c_str()).c_str(), "at");
				if (f)
				{
					mrpt::system::os::fprintf(f, "%u\t%lu\n", step, memUsage);
					mrpt::system::os::fclose(f);
				}
				std::cout << "Ok! (" << ((float)memUsage) / (1024 * 1024) << "Mb)" << std::endl;
			}

			// save the robot estimated pose for each step:
			f_path.printf(
				"%d %f %f %f\n",
				step,
				mapBuilder.getCurrentPoseEstimation()->getMeanVal().x(),
				mapBuilder.getCurrentPoseEstimation()->getMeanVal().y(),
				mapBuilder.getCurrentPoseEstimation()->getMeanVal().yaw()
			);

			f_pathOdo.printf("%i %f %f %f\n", step, odoPose.x(), odoPose.y(), odoPose.phi());

		}  // end of if "rawLogOffset"...

		++step;
		std::cout << "\n---------------- STEP " << step << " | RAWLOG ENTRY " << (unsigned)rawlogEntry << " ----------------" << std::endl;

		// free memory:
		actions.clear_unique();
		observations.clear_unique();
	};

	std::cout << "\n---------------- END!! (total time: " << tictacGlobal.Tac() << " sec) ----------------" << std::endl;

	// save map:
	mapBuilder.getCurrentlyBuiltMap(finalMap);  // create simple map

	std::string str = mrpt::utils::format("%s/_finalmap_.simplemap", options.logOutputDirectoryName.c_str());
	std::cout << "dumping final map in binary format to: " << str << std::endl;
	mapBuilder.saveCurrentMapToFile(str);  // create simple map

	mrpt::slam::CMultiMetricMap *finalPointsMap = mapBuilder.getCurrentlyBuiltMetricMap();
	str = mrpt::utils::format("%s/_finalmaps_.txt", options.logOutputDirectoryName.c_str());
	std::cout << "dumping final metric maps to " << str << "_XXX" << std::endl;
	finalPointsMap->saveMetricMapRepresentationToFile(str);  // create metric maps

	if (win3D)
		win3D->waitForKey();

	//MRPT_TRY_END
}

}  // namespace local
}  // unnamed namespace


void slam_icp()
{
	const bool useRawLogFile = true;
	const std::string rawSensorFileName("mrpt_data\\slam_icp\\dataset_edmonton.rawlog_LASER.txt");
	const std::string rawOdometryFileName("mrpt_data\\slam_icp\\dataset_edmonton.rawlog_ODO.txt");

	const std::string INI_FILENAME("mrpt_data\\slam_icp\\config_slam.ini");
	const std::string INI_SECTION_NAME("MappingApplication");

	if (!mrpt::system::fileExists(INI_FILENAME))
	{
		std::cout << "ini file not found !!!" << std::endl;
		return;
	}

	mrpt::utils::CConfigFile iniFile(INI_FILENAME);

	local::IcpSlamOptions options;

	// load config from file
	//--S [] 2009/08/08: Sang-Wook Lee
	options.rawSensorFileName = rawSensorFileName;
	options.rawOdometryFileName = rawOdometryFileName;
	//--E [] 2009/08/08
	options.rawLogFileName = iniFile.read_string(INI_SECTION_NAME, "rawlog_file", "", /*force existence:*/ true);
	options.rawLogOffset = iniFile.read_int(INI_SECTION_NAME, "rawlog_offset", 0, /*force existence:*/ true);
	options.logOutputDirectoryName = iniFile.read_string(INI_SECTION_NAME, "logOutput_dir", "log_out", /*force existence:*/ true);
	options.logFrequency = iniFile.read_int(INI_SECTION_NAME, "LOG_FREQUENCY", 5, /*force existence:*/ true);
	options.savePoseLog = iniFile.read_bool(INI_SECTION_NAME, "SAVE_POSE_LOG", false, /*force existence:*/ true);

	options.save3DScene = iniFile.read_bool(INI_SECTION_NAME, "SAVE_3D_SCENE", false, /*force existence:*/ true);
	options.camera3DSceneFollowsRobot = iniFile.read_bool(INI_SECTION_NAME, "CAMERA_3DSCENE_FOLLOWS_ROBOT", true, /*force existence:*/ true);

	bool SHOW_PROGRESS_3D_REAL_TIME = false;
	int SHOW_PROGRESS_3D_REAL_TIME_DELAY_MS = 0;
	MRPT_LOAD_CONFIG_VAR(SHOW_PROGRESS_3D_REAL_TIME, bool, iniFile, INI_SECTION_NAME);
	MRPT_LOAD_CONFIG_VAR(SHOW_PROGRESS_3D_REAL_TIME_DELAY_MS, int, iniFile, INI_SECTION_NAME);
	options.showProgress3DRealTime = SHOW_PROGRESS_3D_REAL_TIME;
	options.showProgress3DRealTimeDelayInMillisec = SHOW_PROGRESS_3D_REAL_TIME_DELAY_MS;

	float insertionLinDistance = 0.0f;
	float insertionAngDistance = 0.0f;
	MRPT_LOAD_CONFIG_VAR(insertionLinDistance, float, iniFile, INI_SECTION_NAME);
	MRPT_LOAD_CONFIG_VAR_DEGREES(insertionAngDistance, iniFile, INI_SECTION_NAME);
	options.insertionLinearDistance = insertionLinDistance;
	options.insertionAngularDistance = insertionAngDistance;

	options.metricMapsOptions.loadFromConfigFile(iniFile, INI_SECTION_NAME);
	options.icpOptions.loadFromConfigFile(iniFile, "ICP");
	options.matchAgainstTheGrid = iniFile.read_bool(INI_SECTION_NAME, "matchAgainstTheGrid", true);

	// print params:
	std::cout << "running with the following parameters:" << std::endl;
	std::cout << " RAWLOG file:'" << options.rawLogFileName << "'" << std::endl;
	std::cout << " output directory:\t\t'" << options.logOutputDirectoryName << "'" << std::endl;
	std::cout << " matchAgainstTheGrid:\t\t" << (options.matchAgainstTheGrid ? 'Y' : 'N') << std::endl;
	std::cout << " log record freq:\t\t" << options.logFrequency << std::endl;
	std::cout << "  savePoseLog:\t\t" << (options.savePoseLog ? 'Y' : 'N') << std::endl;
	std::cout << "  save3DScene:\t\t" << (options.save3DScene ? 'Y' : 'N') << std::endl;
	std::cout << "  camera3DSceneFollowsRobot:\t" << (options.camera3DSceneFollowsRobot ? 'Y' : 'N') << std::endl;

	std::cout << std::endl;

	// output options to console
	options.metricMapsOptions.dumpToConsole();
	options.icpOptions.dumpToConsole();

	if (useRawLogFile)
	{
		if (options.rawLogFileName.empty() || !mrpt::system::fileExists(options.rawLogFileName))
		{
			std::cout << "raw log file error !!!" << std::endl;
			return;
		}
	}
	else
	{
		if (options.rawSensorFileName.empty() || !mrpt::system::fileExists(options.rawSensorFileName))
		{
			std::cout << "raw sensor file error !!!" << std::endl;
			return;
		}
		if (options.rawOdometryFileName.empty() || !mrpt::system::fileExists(options.rawOdometryFileName))
		{
			std::cout << "raw odometry file error !!!" << std::endl;
			return;
		}
	}

	//
	std::cout << std::endl << "map building ...." << std::endl;
	local::icp_slam_map_building(options, useRawLogFile);
}
