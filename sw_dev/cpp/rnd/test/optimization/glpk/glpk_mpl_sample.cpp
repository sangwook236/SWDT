//#include "stdafx.h"
#include <glpk.h>
#include <boost/timer/timer.hpp>
#include <iostream>


namespace {
namespace local {

// REF [file] >> ${GLPK_HOME}/examples/mplsamp1.c
void mpl_lp_sample()
{
	glp_prob *lp = glp_create_prob();
	glp_tran *tran = glp_mpl_alloc_wksp();

	int retval = glp_mpl_read_model(tran, "./data/optimization/glpk/egypt.mod", 0);
	if (0 != retval)
	{
		std::cerr << "Error on translating model" << std::endl;
		goto skip;
	}

	retval = glp_mpl_generate(tran, NULL);
	if (0 != retval)
	{
		std::cerr << "Error on generating model" << std::endl;
		goto skip;
	}

	// Convert a model format from MPL to MPS.
	glp_mpl_build_prob(tran, lp);
	retval = glp_write_mps(lp, GLP_MPS_FILE, NULL, "./data/optimization/glpk/egypt.mps");
	if (0 != retval)
	{
		std::cerr << "Error on writing MPS file" << std::endl;
		goto skip;
	}

	// Need to solve the problem.
	// REF [function] >> mpl_mip_sample()

skip:
	glp_mpl_free_wksp(tran);
	glp_delete_prob(lp);
}

// REF [file] >> ${GLPK_HOME}/examples/mplsamp2.c
void mpl_mip_sample()
{
	glp_prob *mip = glp_create_prob();
	glp_tran *tran = glp_mpl_alloc_wksp();

	int retval = glp_mpl_read_model(tran, "./data/optimization/glpk/sudoku.mod", 1);
	if (0 != retval)
	{
		std::cerr << "Error on translating model" << std::endl;
		goto skip;
	}

	retval = glp_mpl_read_data(tran, "./data/optimization/glpk/sudoku.dat");
	if (0 != retval)
	{
		std::cerr << "Error on translating data" << std::endl;
		goto skip;
	}

	retval = glp_mpl_generate(tran, NULL);
	if (0 != retval)
	{
		std::cerr << "Error on generating model" << std::endl;
		goto skip;
	}

	glp_mpl_build_prob(tran, mip);

	// Solve LP problem with the simplex method.
	retval = glp_simplex(mip, NULL);
	if (0 != retval)
	{
		std::cerr << "Error on simplex method." << std::endl;
		goto skip;
	}
	// Solve MIP problem with the branch-and-bound method.
	retval = glp_intopt(mip, NULL);
	if (0 != retval)
	{
		std::cerr << "Error on branch-and-bound method." << std::endl;
		goto skip;
	}

	{
		boost::timer::auto_cpu_timer timer;
		retval = glp_mpl_postsolve(tran, mip, GLP_MIP);
	}
	if (0 != retval)
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
	local::mpl_lp_sample();
	local::mpl_mip_sample();

	// Use AMPL IDE to convert AMPL to MPS.
	//	REF [site] >> http://ampl.com/faqs/how-do-i-write-an-mps-file-for-my-problem/
}

}  // namespace my_glpk
