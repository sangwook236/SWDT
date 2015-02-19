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

// ${OPENSVR_HOME}/JLnkTest/main.cpp.
void test_example()
{
	const std::string infilename("");
	const std::string outfilename("");
	const int MT = 2;  // model type 0 for plane, 1 for line, 2 for vanishing point.
	const int M = 2;  // number of minimal sample sets (MSS).
	double *FirstSamplingVector = NULL;
	const unsigned int NFSamplingType = NFST_NN_ME;
	const double I = 0.05;  // inlier threshold.
	const double SigmaExp = 0.2;

	std::vector<std::vector<float> *> pts;
	{
		std::ifstream ifile(infilename);
		Read_Line(ifile, pts);
	}

	std::vector<std::vector<float> *> *mModels = JlnkSample::run(&pts, M, MT, FirstSamplingVector, NFSamplingType/*, SigmaExp*/);

	std::vector<unsigned int> labels;
	std::vector<unsigned int> labelCount;
	const unsigned int num = JlnkCluster::run(labels, labelCount, &pts, mModels, I, MT);

	{
		std::ofstream ofile(outfilename);
		const unsigned int len = (unsigned int)labels.size();
		for (unsigned int i = 0; i < len; ++i)
		{
			ofile << (*pts[i])[0] << " " << (*pts[i])[1] << " " << (*pts[i])[2] << " " << (*pts[i])[3] << " " << labels.at(i) << std::endl;
		}
		ofile.close();
	}

	for (unsigned int i = 0; i < pts.size(); ++i)
		delete pts[i];

	for (unsigned int i = 0; i < mModels->size(); ++i)
		delete (*mModels)[i];
	delete mModels;
}

}  // namespace local
}  // unnamed namespace

namespace my_jlinkage {

}  // namespace my_jlinkage

int jlinkage_main(int argc, char *argv[])
{
	local::test_example();

	return 0;
}
