//#include "stdafx.h"
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_clustering {

void k_medoids();
void kmeanspp();

}  // namespace my_clustering

int clustering_main(int argc, char *argv[])
{
	//my_clustering::k_medoids();  // not yet implemented

	my_clustering::kmeanspp();

	return 0;
}
