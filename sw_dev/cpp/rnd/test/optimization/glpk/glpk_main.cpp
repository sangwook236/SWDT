//#include "stdafx.h"
#include <glpk.h>
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_glpk {

void simplex_sample();
void interior_point_sample();
void mip_sample();
void mpl_sample();

void netgen_sample();

}  // namespace my_glpk

int glpk_main(int argc, char *argv[])
{
	std::cout << "simplex method ------------------------------------------------------" << std::endl;
	my_glpk::simplex_sample();

	std::cout << "\ninterior-point method -----------------------------------------------" << std::endl;
	my_glpk::interior_point_sample();

	std::cout << "\nmixed integer programming (MIP) -------------------------------------" << std::endl;
	//my_glpk::mip_sample();  // not yet implemented

	std::cout << "\nmathematical programming languag (MPL)-------------------------------" << std::endl;
	my_glpk::mpl_sample();

	std::cout << "\ngraph ---------------------------------------------------------------" << std::endl;
	//my_glpk::netgen_sample();

	return 0;
}
