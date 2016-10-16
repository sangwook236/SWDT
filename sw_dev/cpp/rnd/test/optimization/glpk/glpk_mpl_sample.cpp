//#include "stdafx.h"
#include <glpk.h>
#include <boost/timer/timer.hpp>
#include <iostream>


namespace {
namespace local {

// REF [file] >> ${GLPK_HOME}/examples/mplsamp1.c
void mpl_sample_1()
{
	glp_prob *lp = glp_create_prob();
	glp_tran *tran = glp_mpl_alloc_wksp();

	int ret = glp_mpl_read_model(tran, "./data/optimization/glpk/egypt.mod", 0);
	if (0 != ret)
	{
		std::cerr << "Error on translating model" << std::endl;
		goto skip;
	}

	ret = glp_mpl_generate(tran, NULL);
	if (0 != ret)
	{
		std::cerr << "Error on generating model" << std::endl;
		goto skip;
	}

	glp_mpl_build_prob(tran, lp);
	ret = glp_write_mps(lp, GLP_MPS_FILE, NULL, "./data/optimization/glpk/egypt.mps");
	if (0 != ret)
		std::cerr << "Error on writing MPS file" << std::endl;

skip:
	glp_mpl_free_wksp(tran);
	glp_delete_prob(lp);
}

// REF [file] >> ${GLPK_HOME}/examples/mplsamp2.c
void mpl_sample_2()
{
	glp_prob *mip = glp_create_prob();
	glp_tran *tran = glp_mpl_alloc_wksp();

	int ret = glp_mpl_read_model(tran, "./data/optimization/glpk/sudoku.mod", 1);
	if (0 != ret)
	{
		std::cerr << "Error on translating model" << std::endl;
		goto skip;
	}

	ret = glp_mpl_read_data(tran, "./data/optimization/glpk/sudoku.dat");
	if (0 != ret)
	{
		std::cerr << "Error on translating data" << std::endl;
		goto skip;
	}

	ret = glp_mpl_generate(tran, NULL);
	if (0 != ret)
	{
		std::cerr << "Error on generating model" << std::endl;
		goto skip;
	}

	glp_mpl_build_prob(tran, mip);

	glp_simplex(mip, NULL);
	glp_intopt(mip, NULL);

	{
		boost::timer::auto_cpu_timer timer;
		ret = glp_mpl_postsolve(tran, mip, GLP_MIP);
	}
	if (0 != ret)
		std::cerr << "Error on postsolving model" << std::endl;

skip:
	glp_mpl_free_wksp(tran);
	glp_delete_prob(mip);
}

}  // namespace local
}  // unnamed namespace

namespace my_glpk {

void mpl_sample()
{
	local::mpl_sample_1();
	local::mpl_sample_2();
}

}  // namespace my_glpk
