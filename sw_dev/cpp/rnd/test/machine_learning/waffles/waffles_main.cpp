//include "stdafx.h"
#include <GClasses/GMatrix.h>
#include <GClasses/GVec.h>
#include <GClasses/GManifold.h>
#include <iostream>
#include <stdexcept>


namespace {
namespace local {

void vector_operation()
{
    throw std::runtime_error("not yet implemented");
}

void matrix_operation()
{
    GClasses::GMatrix data;
    data.loadArff("./data/machine_learning/iris.arff");
    //data.swapColumns(1, data.cols() - 1);  // swap the first column with the last one.

    //data.print(std::cout);

    GClasses::GDataColSplitter splitter(data, 1);  // the last 1 column will be used for the label matrix.
    GClasses::GMatrix &features = splitter.features();
    GClasses::GMatrix &labels = splitter.labels();

    //features.print(std::cout);
    //labels.print(std::cout);

    //data.saveArff("./data/machine_learning/waffles/basic_output.arff");
}

}  // namespace local
}  // unnamed namespace

namespace my_waffles {

void ml_example();
void manifold_sculpting_example();

void dimensionality_reduction();

}  // namespace my_waffles

int waffles_main(int argc, char *argv[])
{
    //local::vector_operation();  // not yet implemented.
    local::matrix_operation();

    // REF [site] >>
    //  http://waffles.sourceforge.net/docs/supervised.html
    //  http://waffles.sourceforge.net/docs/new_learner.html

    my_waffles::ml_example();

    // Dimensionality reduction & manifold learning.
    //  GManifoldSculpting.
    //  GIsomap.
    //  GLLE.
    //  GBreadthFirstUnfolding.
    //  GNeuroPCA.
    //  GDynamicSystemStateAligner.
    //  GUnsupervisedBackProp.
    //  GScalingUnfolder.
    my_waffles::manifold_sculpting_example();

    my_waffles::dimensionality_reduction();

	return 0;
}
