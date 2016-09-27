#include "../jlinkage_lib/JLinkage.h"
#include "JlnkCluster.h"
#include "JlnkSample.h"
#include "RandomSampler.h"
#include <iomanip>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>


namespace {
namespace local {

inline void Read_Line(std::istream &in, std::vector<std::vector<float> *> &pts)
{
	while (in)
	{
		std::string str;
		std::getline(in, str, '\n');
		if (0 == str.length()) continue;

		double x0, y0, x1, y1;
		sscanf(str.c_str(), "%lf%*[^0-9-+.eE]%lf%*[^0-9-+.eE]%lf%*[^0-9-+.eE]%lf", &x0, &y0, &x1, &y1);
		double dx = x0 - x1, dy = y0 - y1;
		if (dx*dx + dy*dy < 400)
			continue;

		std::vector<float> *p = new std::vector<float>(4);
		pts.push_back(p);
		(*p)[0] = (float)x0;
		(*p)[1] = (float)y0;
		(*p)[2] = (float)x1;
		(*p)[3] = (float)y1;
	}
	std::cout << "Read Line Done!" << std::endl;
}

// REF [file] >> ${OPENSVR_HOME}/JLnkTest/main.cpp.
void vanishing_point_estimation_from_lines_example()
{
#if 0
	const std::string input_filename("./data/statistics/jlinkage/vp_indoor_lines.in");
	const std::string output_filename(input_filename + ".out");
	const int ModelType = 2;  // Model type 0 for plane, 1 for line, 2 for vanishing point.
#else
	const std::string input_filename("./data/statistics/jlinkage/vp_outdoor_lines.in");
	const std::string output_filename(input_filename + ".out");
	const int ModelType = 2;  // Model type 0 for plane, 1 for line, 2 for vanishing point.
#endif
	const int NumberOfDesiredSamples = 5000;  // Number of minimal sample sets (MSS).
	double *FirstSamplingVector = NULL;
	const unsigned int NFSamplingType = NFST_NN_ME;
	const double InlierThreshold = 2.0;  // Inlier threshold.
	const double SigmaExp = 0.2;

    // Read input.
	std::vector<std::vector<float> *> pts;
	{
        // Input file format : x1 y1 x2 y2 width.
		std::ifstream stream(input_filename.c_str());
		if (!stream)
		{
            std::cerr << "Input file not found : " << input_filename << std::endl;
            return;
		}

		Read_Line(stream, pts);
	}

    // J-linkages.
	std::vector<std::vector<float> *> *models = nullptr;
	std::vector<unsigned int> labels;
	std::vector<unsigned int> labelCount;
	{
		// Sampling.
		models = JlnkSample::run(&pts, NumberOfDesiredSamples, ModelType, FirstSamplingVector, NFSamplingType, SigmaExp);

		// Clustering.
		const unsigned int numClusters = JlnkCluster::run(labels, labelCount, &pts, models, (float)InlierThreshold, ModelType);
		std::cout << "Number of clusters : " << numClusters << std::endl;
	}

    // Write output.
	{
        // Output file format : x1 y1 x2 y2 label.
		std::ofstream stream(output_filename.c_str());
		if (!stream)
		{
            std::cerr << "Output file not found : " << output_filename << std::endl;
            return;
		}

		const unsigned int len = (unsigned int)labels.size();
		for (unsigned int i = 0; i < len; ++i)
			stream << (*pts[i])[0] << ' ' << (*pts[i])[1] << ' ' << (*pts[i])[2] << ' ' << (*pts[i])[3] << ' ' << labels.at(i) << std::endl;
		stream.close();
	}

    // Clean up.
	for (unsigned int i = 0; i < pts.size(); ++i)
		delete pts[i];

	for (unsigned int i = 0; i < models->size(); ++i)
		delete (*models)[i];
	delete models;
}

}  // namespace local
}  // unnamed namespace

namespace my_jlinkage {

}  // namespace my_jlinkage

int jlinkage_main(int argc, char *argv[])
{
	local::vanishing_point_estimation_from_lines_example();

	return 0;
}
