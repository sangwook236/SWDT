#include "stdafx.h"
#include <mrpt/core.h>


#ifdef _DEBUG
//#define new DEBUG_NEW
#endif


namespace {

struct PfLocalizationOptions
{
	mrpt::vector_int particlesList;  // number of initial particles. if size > 1, run the experiments N times.

	std::string rawLogFileName;
	unsigned int rawLogOffset;
	std::string logOutputDirectoryName;

	std::string mapFileName;
	std::string groundTruthFileName;

	int experimentRepetitionCount;

	int scene3dFrequecny;
	bool camera3DSceneFollowsRobot;

	unsigned int experimentTestConvergenceAtStep;
	bool saveStatsOnly;

	bool showProgress3DRealTime;
	int showProgress3DRealTimeDelayInMillisec;

	mrpt::bayes::CParticleFilter::TParticleFilterOptions pfOptions;
	mrpt::poses::CPosePDFParticles::TPredictionParams pdfPredictionOptions;
	mrpt::slam::TSetOfMetricMapInitializers metricMapsOptions;
};

void pf_localization(const PfLocalizationOptions &options)
{
	// load the set of metric maps to consider in the experiments:
	mrpt::slam::CSensFrameProbSequence simpleMap;
	mrpt::slam::CMultiMetricMap metricMap;
	metricMap.setListOfMaps(&options.metricMapsOptions);

	mrpt::random::randomGenerator.randomize();

	// load the map (if any):
	if (!options.mapFileName.empty() && mrpt::utils::fileExists(options.mapFileName))
	{
		// detect file extension:
		const std::string mapExt = mrpt::system::toLowerCase(mrpt::utils::extractFileExtension(options.mapFileName, true));  // Ignore possible .gz extensions

		if (!mapExt.compare("simplemap"))
		{
			// simple map
			std::cout << "loading '.simplemap' file...";
			mrpt::utils::CFileGZInputStream(options.mapFileName) >> simpleMap;
			std::cout << "Ok" << std::endl;

			ASSERT_(simpleMap.size() > 0);

			// build metric map: simple map -> metric map
			std::cout << "building metric map(s) from '.simplemap'...";
			metricMap.loadFromProbabilisticPosesAndObservations(simpleMap);
			std::cout << "Ok" << std::endl;
		}
		else if (!mapExt.compare("gridmap"))
		{
			// grid map
			std::cout << "loading gridmap from '.gridmap'...";
			ASSERT_(metricMap.m_gridMaps.size() == 1);
			mrpt::utils::CFileGZInputStream(options.mapFileName) >> (*metricMap.m_gridMaps[0]);
			std::cout << "Ok" << std::endl;
		}
		else
		{
			std::ostringstream stream;
			stream << "map file has unknown extension: '" << mapExt << "'";
			throw std::logic_error(stream.str());
		}
	}

	// load the rawlog:
	mrpt::slam::CRawlog rawlog;
	std::cout << "opening the rawlog file...";
	rawlog.loadFromRawLogFile(options.rawLogFileName);
	std::cout << "Ok" << std::endl;
	const std::size_t rawlogEntries = rawlog.size();

	// load the ground truth:
	mrpt::math::CMatrixDouble GT(0, 0);
	if (mrpt::utils::fileExists(options.groundTruthFileName) )
	{
		std::cout << "loading ground truth file...";
		GT.loadFromTextFile(options.groundTruthFileName);
		std::cout << "Ok" << std::endl;
	}
	else
		std::cout << "ground truth file: NO" << std::endl;

	// the experiment directory is:
	const std::string overallOutputDirectoryName = mrpt::format("%s_SUMMARY", options.logOutputDirectoryName);

	mrpt::system::createDirectory(overallOutputDirectoryName);
	mrpt::system::deleteFiles(mrpt::format("%s/*.*", overallOutputDirectoryName));

	// create 3D window if requested:
	mrpt::gui::CDisplayWindow3D *win3D = NULL;
	if (options.showProgress3DRealTime)
	{
		win3D = new mrpt::gui::CDisplayWindow3D("PF localization @ MRPT C++ Library (C) 2004-2008", 1000, 600);
		win3D->setCameraZoom(20);
		win3D->setCameraAzimuthDeg(-45);
	}

	// create the 3D scene and get the map only once, later we'll modify only the particles, etc..
	mrpt::opengl::COpenGLScene scene;
	mrpt::slam::COccupancyGridMap2D::TEntropyInfo gridInfo;

	// the gridmap:
	if (metricMap.m_gridMaps.size())
	{
		metricMap.m_gridMaps[0]->computeEntropy(gridInfo);
		std::cout << "the gridmap has " <<  gridInfo.effectiveMappedArea << "m2 observed area, " << (unsigned)gridInfo.effectiveMappedCells << " observed cells" << std::endl;

		{
			mrpt::opengl::CSetOfObjectsPtr plane = mrpt::opengl::CSetOfObjects::Create();
			metricMap.m_gridMaps[0]->getAs3DObject(plane);
			scene.insert(plane);
		}

		if (options.showProgress3DRealTime)
		{
			mrpt::opengl::COpenGLScenePtr ptrScene = win3D->get3DSceneAndLock();

			mrpt::opengl::CSetOfObjectsPtr plane = mrpt::opengl::CSetOfObjects::Create();
			metricMap.m_gridMaps[0]->getAs3DObject(plane);
			ptrScene->insert(plane);

			ptrScene->enableFollowCamera(true);

			win3D->unlockAccess3DScene();
		}
	}

	//
	mrpt::utils::CTicTac tictac, tictacGlobal;
	std::size_t rawlogEntry;
	mrpt::slam::CParticleFilter::TParticleFilterStats PF_stats;

	for (mrpt::vector_int::const_iterator itNum = options.particlesList.begin(); itNum != options.particlesList.end(); ++itNum)
	{
		const int PARTICLE_COUNT = *itNum;
	
		std::cout << std::endl << "-------------------------------------------------------------" << std::endl;
		std::cout << "           running for " << PARTICLE_COUNT << " initial particles" << std::endl;
		std::cout << "-------------------------------------------------------------" << std::endl << std::endl;

		std::cout << "Initial PDF: " << (PARTICLE_COUNT / gridInfo.effectiveMappedArea) << " particles/m2" << std::endl;

		// global stats for all the experiment loops:
		int nConvergenceTests = 0, nConvergenceOK = 0;
		double convergenceTempErrorAccum;
		mrpt::vector_double covergenceErrors;
		covergenceErrors.reserve(options.experimentRepetitionCount);

		// experimental repetitions loop
		tictacGlobal.Tic();
		for (int repetition = 0; repetition < options.experimentRepetitionCount; ++repetition)
		{
			// the experiment directory is:
			const char *OUT_DIR = NULL;
			const char *OUT_DIR_PARTS = NULL;
			const char *OUT_DIR_3D = NULL;
            std::string sOUT_DIR;
            std::string sOUT_DIR_PARTS;
			std::string sOUT_DIR_3D;
		}
	}
}

}  // unnamed namespace

void localization_pf()
{
	const std::string iniFileName(".\\mrpt_data\\localization_pf\\config_localization.ini");
	const std::string iniSectionName("LocalizationExperiment");

	if (!mrpt::utils::fileExists(iniFileName))
	{
		std::cerr << "ini file not found !!!" << std::endl;
		return;
	}

	//
	PfLocalizationOptions options;

	mrpt::utils::CConfigFile iniFile(iniFileName);

	// load configuration:
	// mandatory entries:
	iniFile.read_vector(iniSectionName, "particles_count", mrpt::vector_int(1, 0), options.particlesList, /*fail if not found*/ true);
	options.rawLogFileName = iniFile.read_string(iniSectionName, "rawlog_file", "", /*fail if not found*/ true);
	options.rawLogOffset = iniFile.read_int(iniSectionName, "rawlog_offset", 0);
	options.logOutputDirectoryName = iniFile.read_string(iniSectionName, "logOutput_dir", "", /*fail if not found*/ true);

	// non-mandatory entries:
	options.mapFileName = iniFile.read_string(iniSectionName, "map_file", "");
	options.groundTruthFileName = iniFile.read_string(iniSectionName, "ground_truth_path_file", "");
	options.experimentRepetitionCount = iniFile.read_int(iniSectionName, "experimentRepetitions", 1);
	options.scene3dFrequecny = iniFile.read_int(iniSectionName, "3DSceneFrequency", 10);
	options.camera3DSceneFollowsRobot = iniFile.read_bool(iniSectionName, "3DSceneFollowRobot", true);
	options.experimentTestConvergenceAtStep = iniFile.read_int(iniSectionName, "experimentTestConvergenceAtStep", -1);

	options.saveStatsOnly = iniFile.read_bool(iniSectionName, "SAVE_STATS_ONLY", false);

	options.showProgress3DRealTime = iniFile.read_bool(iniSectionName, "SHOW_PROGRESS_3D_REAL_TIME", false);
	options.showProgress3DRealTimeDelayInMillisec = iniFile.read_int(iniSectionName, "SHOW_PROGRESS_3D_REAL_TIME_DELAY_MS", 1);

	// PF algorithm Options:
	options.pfOptions.loadFromConfigFile(iniFile, "PF_options");

	// PDF Options:
	options.pdfPredictionOptions.KLD_params.loadFromConfigFile(iniFile, "KLD_options");

	// metric map options:
	options.metricMapsOptions.loadFromConfigFile(iniFile, "MetricMap");

	std::cout << std::endl << "-------------------------------------------------------------" << std::endl;
	std::cout << "\t raw log file name = \t" << options.rawLogFileName << std::endl;
	std::cout << "\t map file name = \t" << options.mapFileName << std::endl;
	std::cout << "\t ground truth file name = \t" << options.groundTruthFileName << std::endl;
	std::cout << "\t log output directory name = \t" << options.logOutputDirectoryName << std::endl;
	std::cout << "\t #particles = \t";
	for (mrpt::vector_int::iterator it = options.particlesList.begin(); it != options.particlesList.end(); ++it)
		std::cout << *it << ", ";
	std::cout << std::endl;
	std::cout << "-------------------------------------------------------------" << std::endl;

	options.pfOptions.dumpToConsole();
	options.metricMapsOptions.dumpToConsole();

	//
	std::cout << std::endl << "PF localization ...." << std::endl;
	pf_localization(options);
}
