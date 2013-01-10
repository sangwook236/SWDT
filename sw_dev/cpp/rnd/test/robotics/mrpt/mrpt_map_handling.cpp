//#include "stdafx.h"
#include <mrpt/slam.h>


namespace {
namespace local {

void map_handling_basic()
{
	const std::string simpleMapInputFileName(".\\robotics_data\\mrpt\\localization_pf\\localization_demo.simplemap.gz");
	const std::string simpleMapOutputFileName(".\\robotics_data\\mrpt\\map_handling.simplemap.gz");
	const std::string gridMapInputFileName(".\\robotics_data\\mrpt\\2006-MalagaCampus.gridmap.gz");
	const std::string mapImageFileName(".\\robotics_data\\mrpt\\KAIST-EECS-2ND-INTERSECTION.jpg");
	const std::string metricMapOutputPrefix(".\\robotics_data\\mrpt\\metric_maps");
	const std::string metricMapConfigFileName(".\\robotics_data\\mrpt\\metric_maps_config.ini");
	const std::string metricMapConfigSectionName("MULTI_METRIC_MAP_CONFIGURATION");

	// simple map
	{
		// read simple map
		{
			try
			{
				mrpt::slam::CSensFrameProbSequence simpleMap;
				mrpt::utils::CFileGZInputStream stream(simpleMapInputFileName.c_str());
				std::cout << "reading simple map...";
				stream >> simpleMap;
				std::cout <<"done: " << simpleMap.size() << " observations." << std::endl;
			}
			catch (const std::exception &e)
			{
				std::cerr << "map file loading error: " << e.what() << std::endl;
			}
		}

		// write simple map
		{
			mrpt::slam::CSensFrameProbSequence simpleMap;

			//mrpt::poses::CPose3DPDF posePdf;  // abstract class
			mrpt::poses::CPosePDFGaussian posePdf;
			mrpt::slam::CSensoryFrame observations;
/*
			mrpt::slam::CObservation2DRangeScanPtr observation(new mrpt::slam::CObservation2DRangeScan());
			observations.insert(observation);

			simpleMap.insert(&posePdf, observations);
*/
			//
			if (simpleMap.size() > 0)
			{
				std::cout << "writing simple map...";
				mrpt::utils::CFileGZOutputStream(simpleMapOutputFileName) << simpleMap;
				std::cout << "done." << std::endl;
			}
		}
	}

	// metric map
	{
		bool flag = true;
		mrpt::slam::CSensFrameProbSequence simpleMap;
		try
		{
			mrpt::utils::CFileGZInputStream(simpleMapInputFileName) >> simpleMap;
		}
		catch (const std::exception &e)
		{
			flag = false;
			std::cerr << "map file loading error: " << e.what() << std::endl;
		}

		// read metric maps
		if (flag)
		{
			try
			{
				mrpt::slam::COccupancyGridMap2D gridMap;
				std::cout << "reading grid map...";
				mrpt::utils::CFileGZInputStream(gridMapInputFileName) >> gridMap;
				std::cout <<"done: " << std::endl;
			}
			catch (const std::exception &e)
			{
				std::cerr << "map file loading error: " << e.what() << std::endl;
			}

			const std::string mapImageGridMapFileName(mapImageFileName + std::string(".gridmap.gz"));
			const std::string mapImageMultiMetricMapFileName(mapImageFileName + std::string(".multimetricmap.gz"));

			try
			{
				const float resolution = 0.05f;  // the size of a pixel (cell), [meters]
				const float xCentralPixel = -1.0f;
				const float yCentralPixel = -1.0f;

				mrpt::slam::COccupancyGridMap2D gridMap;
				std::cout << "reading grid map from image file...";
				gridMap.loadFromBitmapFile(mapImageFileName, resolution, xCentralPixel, yCentralPixel);
				//mrpt::utils::CMRPTImage img;
				//img.loadFromFile(mapImageFileName), 
				//gridMap.loadFromBitmap(img, resolution, xCentralPixel, yCentralPixel);
				std::cout << "done!: " << (gridMap.getXMax() - gridMap.getXMin()) << " x " << (gridMap.getYMax() - gridMap.getYMin()) << " m" << std::endl;

				std::cout << "writing grid map...";
				mrpt::utils::CFileGZOutputStream(mapImageGridMapFileName) << gridMap;
				std::cout << "done." << std::endl;
			}
			catch (const std::exception &e)
			{
				std::cerr << "map file loading error: " << e.what() << std::endl;
			}

			try
			{
				//mrpt::slam::CMetricMap metricMap;  // abstract class
				mrpt::slam::CMultiMetricMap metricMap;

				//std::cout << "reading metric map...";
				//mrpt::utils::CFileGZInputStream(mapImageGridMapFileName) >> metricMap;  // compile error !!!
				//std::cout <<"done: " << mapImageGridMapFileName << std::endl;

				std::cout << "reading grid map...";
				mrpt::slam::COccupancyGridMap2DPtr gridMap(new mrpt::slam::COccupancyGridMap2D());
				mrpt::utils::CFileGZInputStream(mapImageGridMapFileName) >> *gridMap;
				//metricMap.m_pointsMaps.push_back();
				//metricMap.m_landmarksMap = ;
				//metricMap.m_beaconMap = ;
				metricMap.m_gridMaps.push_back(gridMap);
				//metricMap.m_gasGridMaps.push_back();
				//metricMap.m_heightMaps.push_back();
				//metricMap.m_colourPointsMap = ;
				std::cout <<"done: " << mapImageGridMapFileName << std::endl;

				std::cout << "writing metric map...";
				mrpt::utils::CFileGZOutputStream(mapImageMultiMetricMapFileName) << metricMap;
				std::cout <<"done: " << mapImageMultiMetricMapFileName << std::endl;

				mrpt::slam::CMultiMetricMap metricMap2;
				std::cout << "reading metric map...";
				mrpt::utils::CFileGZInputStream(mapImageMultiMetricMapFileName) >> metricMap2;
				std::cout <<"done: " << mapImageMultiMetricMapFileName << std::endl;
			}
			catch (const std::exception &e)
			{
				std::cerr << "map file loading error: " << e.what() << std::endl;
			}
		}

		// write metric maps
		if (flag)
		{
			mrpt::slam::TSetOfMetricMapInitializers mapOptions;
			mapOptions.loadFromConfigFile(mrpt::utils::CConfigFile(metricMapConfigFileName), metricMapConfigSectionName);

			//mrpt::slam::CMetricMap metricMap;  // abstract class
			mrpt::slam::CMultiMetricMap metricMap;
			metricMap.setListOfMaps(&mapOptions);
/*
			mrpt::slam::CObservation *observation;
			mrpt::poses::CPose3D pose;
			metricMap.insertObservation(&observation, &pose);

			const double logLikehood = computeObservationLikelihood(&observation, pose);
*/
			// build metric maps:
			std::cout << "building metric maps...";
			metricMap.loadFromProbabilisticPosesAndObservations(simpleMap);
			std::cout << "done." << std::endl;

			// save metric maps:
			metricMap.saveMetricMapRepresentationToFile(metricMapOutputPrefix);

			// grid maps:
			for (size_t i = 0; i < metricMap.m_gridMaps.size(); ++i)
			{
				const std::string str = mrpt::format("%s_gridmap_no%02u.gridmap.gz", metricMapOutputPrefix.c_str(), (unsigned)i);
				std::cout << "saving gridmap #" << i << " to " << str;

				mrpt::utils::CFileGZOutputStream stream(str);
				stream << *metricMap.m_gridMaps[i];

				std::cout << " done." << std::endl;
			}
		}
	}
}

void map_handling__bitmap2pointmap()
{
	const std::string mapImageFileName(".\\robotics_data\\mrpt\\PIRO_1st_floor_map_GC2009_final.bmp");
	const std::string metricMapOutputPrefix(mapImageFileName + ".metric_maps");

	//
	const float resolution = 0.01f;  // the size of a pixel (cell), [meters]
	const float xCentralPixel = 0.0f;
	const float yCentralPixel = 0.0f;

	//
	mrpt::slam::COccupancyGridMap2DPtr gridMap(new mrpt::slam::COccupancyGridMap2D());
	std::cout << "reading grid map from image file...";
	gridMap->loadFromBitmapFile(mapImageFileName, resolution, xCentralPixel, yCentralPixel);
	std::cout << "done!: " << (gridMap->getXMax() - gridMap->getXMin()) << " x " << (gridMap->getYMax() - gridMap->getYMin()) << " m" << std::endl;

	std::cout << "converting grid map to metric map...";
	mrpt::slam::CMultiMetricMap metricMap;
	metricMap.m_gridMaps.push_back(gridMap);
	std::cout << " done." << std::endl;

	// save metric maps:
	std::cout << "writing grid map to metric map...";
	metricMap.saveMetricMapRepresentationToFile(metricMapOutputPrefix);
	std::cout << " done." << std::endl;
}

}  // namespace local
}  // unnamed namespace

namespace my_mrpt {

void map_handling()
{
	//local::map_handling_basic();
	local::map_handling__bitmap2pointmap();
}

}  // namespace my_mrpt
