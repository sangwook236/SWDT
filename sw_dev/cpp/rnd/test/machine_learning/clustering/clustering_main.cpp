//#include "stdafx.h"
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace clustering {

void k_means();
void k_medoids();

}  // namespace clustering

int clustering_main(int argc, char *argv[])
{
	clustering::k_means();  // not yet implemented
	clustering::k_medoids();  // not yet implemented

	return 0;
}
