//#include "stdafx.h"
#include <glpk.h>
#include <boost/timer/timer.hpp>
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_glpk {

// [ref] ${GLPK_HOME}/examples/iptsamp.c
void interior_point_sample()
{
	glp_prob *P = glp_create_prob();

	glp_read_mps(P, GLP_MPS_DECK, NULL, "./data/optimization/glpk/25fv47.mps");
	
	{
		boost::timer::auto_cpu_timer timer;
		const int retval = glp_interior(P, NULL);
	}
	
	glp_print_ipt(P, "./data/optimization/glpk/25fv47_ipt.txt");
	
	glp_delete_prob(P);
	P = NULL;
}

}  // namespace my_glpk
