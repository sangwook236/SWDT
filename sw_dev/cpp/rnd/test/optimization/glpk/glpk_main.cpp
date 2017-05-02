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
	// Information on solution.
	//	REF [site] >> https://en.wikibooks.org/wiki/GLPK/Solution_information

	std::cout << "Simplex method ------------------------------------------------------" << std::endl;
	//my_glpk::simplex_sample();

	std::cout << "\nInterior-point method -----------------------------------------------" << std::endl;
	//my_glpk::interior_point_sample();

	std::cout << "\nMixed integer programming (MIP) -------------------------------------" << std::endl;
	//my_glpk::mip_sample();

	std::cout << "\nMathematical programming language (MPL) -----------------------------" << std::endl;
	//	- GMPL.
	my_glpk::mpl_sample();

	std::cout << "\nGraph ---------------------------------------------------------------" << std::endl;
	//my_glpk::netgen_sample();

	return 0;
}
