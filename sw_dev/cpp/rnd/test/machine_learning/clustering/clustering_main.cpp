//#include "stdafx.h"
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_clustering {

void k_means();
void k_medoids();

}  // namespace my_clustering

int clustering_main(int argc, char *argv[])
{
	my_clustering::k_means();  // not yet implemented
	my_clustering::k_medoids();  // not yet implemented

	return 0;
}
