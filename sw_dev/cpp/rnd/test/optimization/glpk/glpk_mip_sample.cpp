//#include "stdafx.h"
#include <glpk.h>
#include <iostream>


namespace {
namespace local{

// REF [site] >> https://gist.github.com/msakai/2450935
/*
Maximize
	obj: x1 + 2 x2 + 3 x3 + x4
Subject to
	c1: -x1 + x2 + x3 + 10 x4 <= 20
	c2: x1 - 3 x2 + x3 <= 30
	c3: x2 - 3.5 x4 = 0
Bounds
	0 <= x1 <= 40
	2 <= x4 <= 3
General
	x4
End
*/
void mip_simple_sample()
{
	glp_prob *mip = glp_create_prob();
	glp_set_prob_name(mip, "mip-simple-sample");
	glp_set_obj_dir(mip, GLP_MAX);

	// Objective.
	glp_add_cols(mip, 4);
	glp_set_col_name(mip, 1, "x1");
	glp_set_col_bnds(mip, 1, GLP_DB, 0.0, 40.0);
	glp_set_obj_coef(mip, 1, 1.0);
	glp_set_col_name(mip, 2, "x2");
	glp_set_col_bnds(mip, 2, GLP_LO, 0.0, 0.0);
	glp_set_obj_coef(mip, 2, 2.0);
	glp_set_col_name(mip, 3, "x3");
	glp_set_col_bnds(mip, 3, GLP_LO, 0.0, 0.0);
	glp_set_obj_coef(mip, 3, 3.0);
	glp_set_col_name(mip, 4, "x4");
	glp_set_col_bnds(mip, 4, GLP_DB, 2.0, 3.0);
	glp_set_obj_coef(mip, 4, 1.0);
	glp_set_col_kind(mip, 4, GLP_IV);  // Integer variable.

	// Constraints.
	glp_add_rows(mip, 3);
	glp_set_row_name(mip, 1, "c1");
	glp_set_row_bnds(mip, 1, GLP_DB, 0.0, 20.0);
	glp_set_row_name(mip, 2, "c2");
	glp_set_row_bnds(mip, 2, GLP_DB, 0.0, 30.0);
	glp_set_row_name(mip, 3, "c3");
	glp_set_row_bnds(mip, 3, GLP_FX, 0.0, 0);

	int ia[1 + 9] = { 0, }, ja[1 + 9] = { 0, };
	double ar[1 + 9] = { 0.0, };
	ia[1] = 1, ja[1] = 1, ar[1] = -1;   // a[1,1] = -1.
	ia[2] = 1, ja[2] = 2, ar[2] = 1;    // a[1,2] = 1.
	ia[3] = 1, ja[3] = 3, ar[3] = 1;    // a[1,3] = 1.
	ia[4] = 1, ja[4] = 4, ar[4] = 10;   // a[1,4] = 10.
	ia[5] = 2, ja[5] = 1, ar[5] = 1;    // a[2,1] = 1.
	ia[6] = 2, ja[6] = 2, ar[6] = -3;   // a[2,2] = -3.
	ia[7] = 2, ja[7] = 3, ar[7] = 1;    // a[2,3] = 1.
	ia[8] = 3, ja[8] = 2, ar[8] = 1;    // a[3,2] = 1.
	ia[9] = 3, ja[9] = 4, ar[9] = -3.5; // a[3,4] = -3.5.
	glp_load_matrix(mip, 9, ia, ja, ar);

	// Solve.
	glp_iocp parm;
	glp_init_iocp(&parm);
	parm.presolve = GLP_ON;
	// Solve MIP problem with the branch-and-bound method.
	const int retval = glp_intopt(mip, &parm);
	if (0 != retval)
	{
		std::cerr << "Error on branch-and-bound method." << std::endl;
		glp_delete_prob(mip);
		return;
	}

	// Output.
	const double z = glp_mip_obj_val(mip);
	const double x1 = glp_mip_col_val(mip, 1);
	const double x2 = glp_mip_col_val(mip, 2);
	const double x3 = glp_mip_col_val(mip, 3);
	const double x4 = glp_mip_col_val(mip, 4);
	std::cout << "\nz = " << z << "; x1 = " << x1 << ", x2 = " << x2 << ", x3 = " << x3 << ", x4 = " << x4 << std::endl;;
	// z = 122.5; x1 = 40, x2 = 10.5, x3 = 19.5, x4 = 3.

	// Clear up.
	glp_delete_prob(mip);
}

// REF [site] >> https://lists.gnu.org/archive/html/help-glpk/2009-05/msg00077.html
void mip_mps_sample()
{
	glp_prob *mip = glp_create_prob();

	int retval = glp_read_mps(mip, GLP_MPS_FILE, NULL, "./data/optimization/p0033.mps");
	//int retval = glp_read_mps(mip, GLP_MPS_FILE, NULL, "./data/optimization/flugpl.mps");
	if (0 != retval)
	{
		std::cerr << "Model file not found." << std::endl;
		glp_delete_prob(mip);
		return;
	}

	glp_adv_basis(mip, NULL);
	// Solve LP problem with the simplex method.
	retval = glp_simplex(mip, NULL);
	if (0 != retval)
	{
		std::cerr << "Error on simplex method." << std::endl;
		glp_delete_prob(mip);
		return;
	}
	// Solve MIP problem with the branch-and-bound method.
	retval = glp_intopt(mip, NULL);
	if (0 != retval)
	{
		std::cerr << "Error on branch-and-bound method." << std::endl;
		glp_delete_prob(mip);
		return;
	}

	glp_print_mip(mip, "./data/optimization/glpk/p0033.txt");
	//glp_print_mip(mip, "./data/optimization/glpk/flugpl.txt");

	glp_delete_prob(mip);
}

void path_cover_problem()
{
	// Use AMPL IDE to convert AMPL to MPS.
	//	REF [site] >> http://ampl.com/faqs/how-do-i-write-an-mps-file-for-my-problem/

	glp_prob *mip = glp_create_prob();

	// Read MPS file format.
	//	- RDA data: 2016/04/06, adaptor 1, side 0deg.
	int retval = glp_read_mps(mip, GLP_MPS_FILE, NULL, "./data/optimization/path_cover_problem.mps");
	if (0 != retval)
	{
		std::cerr << "Model file not found." << std::endl;
		glp_delete_prob(mip);
		return;
	}

	glp_adv_basis(mip, NULL);
	// Solve LP problem with the simplex method.
	retval = glp_simplex(mip, NULL);
	if (0 != retval)
	{
		std::cerr << "Error on simplex method." << std::endl;
		glp_delete_prob(mip);
		return;
	}
	// Solve MIP problem with the branch-and-bound method.
	retval = glp_intopt(mip, NULL);
	if (0 != retval)
	{
		std::cerr << "Error on branch-and-bound method." << std::endl;
		glp_delete_prob(mip);
		return;
	}

	glp_print_mip(mip, "./data/optimization/glpk/path_cover_problem_mip.txt");
	//glp_print_sol(mip, "./data/optimization/glpk/path_cover_problem_sol.txt");

	glp_delete_prob(mip);
}

}  // namespace local
}  // unnamed namespace

namespace my_glpk {

void mip_sample()
{
	// REF [function] >> local::mpl_sample_2() in ./glpk_mpl_sample.cpp

	// Sample ---------------------------------------------
	//local::mip_simple_sample();
	//local::mip_mps_sample();

	// Application ----------------------------------------
	// Path cover problem.
	local::path_cover_problem();
}

}  // namespace my_glpk
