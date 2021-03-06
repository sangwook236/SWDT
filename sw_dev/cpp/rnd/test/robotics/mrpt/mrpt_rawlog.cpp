//#include "stdafx.h"
#include <mrpt/slam.h>
#include <mrpt/utils/CFileGZInputStream.h>
#include <mrpt/system/filesystem.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_mrpt {

void rawlog()
{
	const std::string rawLogFileName = "data/robotics/mrpt/test_dataset.rawlog";

	// write to raw log file
	{
		mrpt::utils::CFileOutputStream stream(rawLogFileName);

		mrpt::slam::CActionRobotMovement2D::TMotionModelOptions motionModelOptions;
		motionModelOptions.modelSelection = mrpt::slam::CActionRobotMovement2D::mmGaussian;
		motionModelOptions.gausianModel.a1 = 0.01f;  // ratio of motion to x/y std.dev [meters/meter]
		motionModelOptions.gausianModel.a2 = 0.001f * (180.0f / M_PIf);  // ratio of motion to phi std.dev [meters/degree]
		motionModelOptions.gausianModel.a3 = mrpt::utils::DEG2RAD(1.0f);  // ratio of rotation to x/y std.dev [degrees/meter]
		motionModelOptions.gausianModel.a4 = 0.05f;  // ratio of rotation to phi std.dev [degrees/degree]
		motionModelOptions.gausianModel.minStdXY = 0.01f;  // minimum std.dev of x/y [meters]
		motionModelOptions.gausianModel.minStdPHI = mrpt::utils::DEG2RAD(0.2f);  // minimum std.dev of phi [degrees]

		const size_t rawDataCount = 1000;
		const size_t sensorDataCount = 100;

		size_t step = 0;
		while (++step <= rawDataCount)
		{
			mrpt::slam::CActionCollection actions;
			mrpt::slam::CSensoryFrame sensorFrame;

			// fill out the actions
			mrpt::slam::CActionRobotMovement2D anAction;  // for example, 2D odometry
			anAction.timestamp = mrpt::system::getCurrentTime();
			anAction.estimationMethod = mrpt::slam::CActionRobotMovement2D::emOdometry;
			anAction.hasEncodersInfo = false;
			anAction.hasVelocities = false;
			//anAction.motionModelConfiguration = motionModelOptions;

			const float x = std::rand();
			const float y = std::rand();
			const float phi = std::rand();
			mrpt::poses::CPose2D pose(x, y, phi);
			//pose.x(x);
			//pose.y(y);
			//pose.phi(phi);
			anAction.computeFromOdometry(pose, motionModelOptions);

			actions.insert(anAction);

			// fill out the observations
			// create a smart pointer with an empty observation
			mrpt::slam::CObservation2DRangeScanPtr anObservation(new mrpt::slam::CObservation2DRangeScan());
			anObservation->timestamp = mrpt::system::getCurrentTime();
			anObservation->scan.reserve(sensorDataCount);
			anObservation->validRange.reserve(sensorDataCount);
			anObservation->aperture = M_PIf;  // [rad]
			anObservation->rightToLeft = true;
			anObservation->maxRange = 30.0f;  // [m]
			//anObservation->sensorPose = ;
			//anObservation->stdError = ;
			//anObservation->beamAperture = ;
			//anObservation->deltaPitch = ;

			for (size_t i = 0; i < sensorDataCount; ++i)
			{
				const float range = std::rand();
				anObservation->scan.push_back(range);
				anObservation->validRange.push_back(range <= anObservation->maxRange);
			}

			sensorFrame.insert(anObservation);  // 'anObservation' will be automatically freed

			// save to the rawlog file
			stream << actions << sensorFrame;
		}
	}

	// read from raw log file
	{
		if (!mrpt::system::fileExists(rawLogFileName))
		{
			std::cout << "raw log file error !!!" << std::endl;
			return;
		}

		mrpt::slam::CActionCollectionPtr actions;
		mrpt::slam::CSensoryFramePtr observations;

		mrpt::utils::CFileGZInputStream rawlogFile(rawLogFileName);

		using mrpt::slam::CActionRobotMovement2D;
		using mrpt::slam::CObservation2DRangeScan;

		mrpt::poses::CPose2D pose(0, 0, 0);
		std::size_t rawlogEntry = 0;
		while (mrpt::slam::CRawlog::readActionObservationPair(rawlogFile, actions, observations, rawlogEntry))
		{
			for (mrpt::slam::CActionCollection::iterator it = actions->begin(); it != actions->end(); ++it)
			{
				//if ((*it)->GetRuntimeClass() == CLASS_ID(CAction))
				if ((*it)->GetRuntimeClass() == CLASS_ID(CActionRobotMovement2D))
				{
					//mrpt::poses::CPose2D pose;
					//((mrpt::slam::CActionRobotMovement2DPtr)*it)->drawSingleSample(pose);
					//pose = ((mrpt::slam::CActionRobotMovement2DPtr)*it)->poseChange->getMeanVal();
					pose = pose + ((mrpt::slam::CActionRobotMovement2DPtr)*it)->poseChange->getMeanVal();

					//std::cout << '#' << rawlogEntry << ": " << pose.x() << ", " << pose.y() << ", " << pose.phi() << std::endl;
				}
			}

			for (mrpt::slam::CSensoryFrame::iterator it = observations->begin(); it != observations->end(); ++it)
			{
				//if ((*it)->GetRuntimeClass() == CLASS_ID(CObservation))
				if ((*it)->GetRuntimeClass() == CLASS_ID(CObservation2DRangeScan))
				{
					const mrpt::vector_float &scan = ((mrpt::slam::CObservation2DRangeScanPtr)*it)->scan;

					//std::cout << '#' << rawlogEntry << ": ";
					//for (mrpt::vector_float::const_iterator it = scan.begin(); it != scan.end(); ++it)
					//	std::cout << *it << ", ";
					//std::cout << std::endl;
				}
			}

			actions.clear_unique();
			observations.clear_unique();
		}
	}
}

}  // namespace my_mrpt
