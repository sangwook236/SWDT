//#include "stdafx.h"
#include <glpk.h>
#include <boost/timer/timer.hpp>
#include <iostream>


namespace {
namespace local {

// REF [file] >> ${GLPK_HOME}/examples/sample.c
void simple_simplex_sample()
{
	glp_prob *lp = glp_create_prob();

	glp_set_prob_name(lp, "sample");
	glp_set_obj_dir(lp, GLP_MAX);

	glp_add_rows(lp, 3);
	glp_set_row_name(lp, 1, "p");
	glp_set_row_bnds(lp, 1, GLP_UP, 0.0, 100.0);
	glp_set_row_name(lp, 2, "q");
	glp_set_row_bnds(lp, 2, GLP_UP, 0.0, 600.0);
	glp_set_row_name(lp, 3, "r");
	glp_set_row_bnds(lp, 3, GLP_UP, 0.0, 300.0);
	glp_add_cols(lp, 3);
	glp_set_col_name(lp, 1, "x1");
	glp_set_col_bnds(lp, 1, GLP_LO, 0.0, 0.0);
	glp_set_obj_coef(lp, 1, 10.0);
	glp_set_col_name(lp, 2, "x2");
	glp_set_col_bnds(lp, 2, GLP_LO, 0.0, 0.0);
	glp_set_obj_coef(lp, 2, 6.0);
	glp_set_col_name(lp, 3, "x3");
	glp_set_col_bnds(lp, 3, GLP_LO, 0.0, 0.0);

	glp_set_obj_coef(lp, 3, 4.0);

	int ia[1 + 1000], ja[1 + 1000];
	double ar[1 + 1000];
	ia[1] = 1, ja[1] = 1, ar[1] =  1.0;  // a[1,1] =  1.
	ia[2] = 1, ja[2] = 2, ar[2] =  1.0;  // a[1,2] =  1.
	ia[3] = 1, ja[3] = 3, ar[3] =  1.0;  // a[1,3] =  1.
	ia[4] = 2, ja[4] = 1, ar[4] = 10.0;  // a[2,1] = 10.
	ia[5] = 3, ja[5] = 1, ar[5] =  2.0;  // a[3,1] =  2.
	ia[6] = 2, ja[6] = 2, ar[6] =  4.0;  // a[2,2] =  4
	ia[7] = 3, ja[7] = 2, ar[7] =  2.0;  // a[3,2] =  2.
	ia[8] = 2, ja[8] = 3, ar[8] =  5.0;  // a[2,3] =  5.
	ia[9] = 3, ja[9] = 3, ar[9] =  6.0;  // a[3,3] =  6.
	glp_load_matrix(lp, 9, ia, ja, ar);

	{
		boost::timer::auto_cpu_timer timer;
		const int retval = glp_simplex(lp, NULL);
	}

	const double z = glp_get_obj_val(lp);
	const double x1 = glp_get_col_prim(lp, 1);
	const double x2 = glp_get_col_prim(lp, 2);
	const double x3 = glp_get_col_prim(lp, 3);
	std::cout << "\nz = " << z << ", x1 = " << x1 << ", x2 = " << x2 << ", x3 = " << x3 << std::endl;

	glp_delete_prob(lp);
	lp = NULL;
}

// REF [file] >> ${GLPK_HOME}/examples/spxsamp1.c
void simplex_sample_1()
{
	glp_prob *P = glp_create_prob();

	glp_read_mps(P, GLP_MPS_DECK, NULL, "./data/optimization/glpk/25fv47.mps");
	glp_adv_basis(P, 0);

	{
		boost::timer::auto_cpu_timer timer;
		const int retval = glp_simplex(P, NULL);
	}

	glp_print_sol(P, "./data/optimization/glpk/25fv47_simplex_1.txt");

	glp_delete_prob(P);
	P = NULL;
}

// REF [file] >> ${GLPK_HOME}/examples/spxsamp2.c
void simplex_sample_2()
{
	glp_prob *P = glp_create_prob();

	glp_read_mps(P, GLP_MPS_DECK, NULL, "./data/optimization/glpk/25fv47.mps");
	glp_smcp parm;
	glp_init_smcp(&parm);
	parm.meth = GLP_DUAL;

	{
		boost::timer::auto_cpu_timer timer;
		const int retval = glp_simplex(P, &parm);
	}

	glp_print_sol(P, "./data/optimization/glpk/25fv47_simplex_2.txt");

	glp_delete_prob(P);
	P = NULL;
}

}  // namespace local
}  // unnamed namespace

namespace my_glpk {

void simplex_sample()
{
	local::simple_simplex_sample();
	local::simplex_sample_1();
	local::simplex_sample_2();
}

}  // namespace my_glpk
