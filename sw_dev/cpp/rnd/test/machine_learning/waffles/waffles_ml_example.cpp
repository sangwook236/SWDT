//#include "stdafx.h"
#include <GClasses/GApp.h>
#include <GClasses/GError.h>
#include <GClasses/GDecisionTree.h>
#include <GClasses/GNeuralNet.h>
#include <GClasses/GActivation.h>
#include <GClasses/GKNN.h>
#include <GClasses/GNaiveBayes.h>
#include <GClasses/GEnsemble.h>
#include <iostream>


namespace {
namespace local {

void decision_tree(GClasses::GMatrix &features, GClasses::GMatrix &labels, const GClasses::GVec &test_features, GClasses::GVec &predicted_labels)
{
	GClasses::GDecisionTree model;
	model.train(features, labels);
	model.predict(test_features, predicted_labels);
}

void neural_network(GClasses::GMatrix &features, GClasses::GMatrix &labels, const GClasses::GVec &test_features, GClasses::GVec &predicted_labels)
{
	GClasses::GNeuralNet *pNN = new GClasses::GNeuralNet();
	pNN->addLayer(new GClasses::GLayerClassic(FLEXIBLE_SIZE, 3));
	pNN->addLayer(new GClasses::GLayerClassic(3, FLEXIBLE_SIZE));
	pNN->setLearningRate(0.1);
	pNN->setMomentum(0.1);
	GClasses::GAutoFilter af(pNN);
	af.train(features, labels);
	af.predict(test_features, predicted_labels);
}

void knn(GClasses::GMatrix &features, GClasses::GMatrix &labels, const GClasses::GVec &test_features, GClasses::GVec &predicted_labels)
{
	GClasses::GKNN model;
	model.setNeighborCount(3);  // use the 3-nearest neighbors
	model.setInterpolationMethod(GClasses::GKNN::Linear); // use linear interpolation
	model.train(features, labels);
	model.predict(test_features, predicted_labels);
}

void naivebayes(GClasses::GMatrix &features, GClasses::GMatrix &labels, const GClasses::GVec &test_features, GClasses::GVec &predicted_labels)
{
	GClasses::GAutoFilter model(new GClasses::GNaiveBayes());
	model.train(features, labels);
	model.predict(test_features, predicted_labels);
}

void ensemble(GClasses::GMatrix &features, GClasses::GMatrix &labels, const GClasses::GVec &test_features, GClasses::GVec &predicted_labels)
{
	GClasses::GBag ensemble;
	for(size_t i = 0; i < 50; i++)
	{
		GClasses::GDecisionTree *pDT = new GClasses::GDecisionTree();
		pDT->useRandomDivisions(1); // Make random tree
		ensemble.addLearner(pDT);
	}
	ensemble.train(features, labels);
	ensemble.predict(test_features, predicted_labels);
}

}  // namespace local
}  // unnamed namespace

namespace my_waffles {

// REF [file] >> ${WAFFLES_HOME}/demos/hello_ml/src/main.cpp.
void ml_example()
{
	// Define the feature attributes (or columns)
	std::vector<size_t> feature_values;
	feature_values.push_back(0);  // diameter = continuous
	feature_values.push_back(3);  // crust_type = { thin_crust=0, Chicago_style_deep_dish=1, Neapolitan=2 }
	feature_values.push_back(2);  // meatiness = { vegan=0, meaty=1 }
	feature_values.push_back(4);  // presentation = { dine_in=0, take_out=1, delivery=2, frozen=3 }

	// Define the label attributes (or columns)
	std::vector<size_t> label_values;
	label_values.push_back(2);  // taste = { lousy=0, delicious=1 }
	label_values.push_back(0);  // cost = continuous

	// Make some contrived hard-coded training data
	GClasses::GMatrix feat(feature_values);
	feat.newRows(12);
	GClasses::GMatrix lab(label_values);
	lab.newRows(12);
	//        diameter            crust        meatiness     presentation           taste               cost
	feat[0][0] = 14.0;  feat[0][1] = 1;  feat[0][2] = 1;  feat[0][3] = 0;   lab[0][0] = 1;  lab[0][1] = 22.95;
	feat[1][0] = 12.0;  feat[1][1] = 0;  feat[1][2] = 0;  feat[1][3] = 3;   lab[1][0] = 0;  lab[1][1] = 3.29;
	feat[2][0] = 14.0;  feat[2][1] = 1;  feat[2][2] = 1;  feat[2][3] = 2;   lab[2][0] = 1;  lab[2][1] = 15.49;
	feat[3][0] = 12.0;  feat[3][1] = 2;  feat[3][2] = 0;  feat[3][3] = 0;   lab[3][0] = 1;  lab[3][1] = 16.65;
	feat[4][0] = 18.0;  feat[4][1] = 1;  feat[4][2] = 1;  feat[4][3] = 3;   lab[4][0] = 0;  lab[4][1] = 9.99;
	feat[5][0] = 14.0;  feat[5][1] = 1;  feat[5][2] = 1;  feat[5][3] = 0;   lab[5][0] = 1;  lab[5][1] = 14.49;
	feat[6][0] = 12.0;  feat[6][1] = 2;  feat[6][2] = 0;  feat[6][3] = 2;   lab[6][0] = 1;  lab[6][1] = 19.65;
	feat[7][0] = 14.0;  feat[7][1] = 0;  feat[7][2] = 1;  feat[7][3] = 1;   lab[7][0] = 0;  lab[7][1] = 6.99;
	feat[8][0] = 14.0;  feat[8][1] = 1;  feat[8][2] = 1;  feat[8][3] = 2;   lab[8][0] = 1;  lab[8][1] = 19.95;
	feat[9][0] = 14.0;  feat[9][1] = 2;  feat[9][2] = 0;  feat[9][3] = 3;   lab[9][0] = 0;  lab[9][1] = 12.99;
	feat[10][0] = 16.0; feat[10][1] = 0; feat[10][2] = 1; feat[10][3] = 0;  lab[10][0] = 0; lab[10][1] = 12.20;
	feat[11][0] = 14.0; feat[11][1] = 1; feat[11][2] = 1; feat[11][3] = 1;  lab[11][0] = 1; lab[11][1] = 15.01;

	// Make a test vector
	GClasses::GVec test_features(4);
	GClasses::GVec predicted_labels(2);
	std::cout << "This demo trains and tests several supervised learning models using some contrived hard-coded training data to predict the tastiness and cost of a pizza.\n\n";
	test_features[0] = 15.0; test_features[1] = 2; test_features[2] = 0; test_features[3] = 0;
	std::cout << "Predicting labels for a 15 inch pizza with a Neapolitan-style crust, no meat, for dine-in.\n\n";

	// Use several models to make predictions
	std::cout.precision(4);
	local::decision_tree(feat, lab, test_features, predicted_labels);
	std::cout << "The decision tree predicts the taste is " << (predicted_labels[0] == 0 ? "lousy" : "delicious") << ", and the cost is $" << predicted_labels[1] << std::endl;

	local::neural_network(feat, lab, test_features, predicted_labels);
	std::cout << "The neural network predicts the taste is " << (predicted_labels[0] == 0 ? "lousy" : "delicious") << ", and the cost is $" << predicted_labels[1] << std::endl;

	local::knn(feat, lab, test_features, predicted_labels);
	std::cout << "The knn model predicts the taste is " << (predicted_labels[0] == 0 ? "lousy" : "delicious") << ", and the cost is $" << predicted_labels[1] << std::endl;

	local::naivebayes(feat, lab, test_features, predicted_labels);
	std::cout << "The naive Bayes model predicts the taste is " << (predicted_labels[0] == 0 ? "lousy" : "delicious") << ", and the cost is $" << predicted_labels[1] << std::endl;

	local::ensemble(feat, lab, test_features, predicted_labels);
	std::cout << "Random forest predicts the taste is " << (predicted_labels[0] == 0 ? "lousy" : "delicious") << ", and the cost is $" << predicted_labels[1] << std::endl;
}

}  // namespace my_waffles
