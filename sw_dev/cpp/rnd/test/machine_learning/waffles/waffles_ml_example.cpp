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
	model.setNeighborCount(3); // use the 3-nearest neighbors
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
	feature_values.push_back(0); // diameter = continuous
	feature_values.push_back(3); // crust_type = { thin_crust=0, Chicago_style_deep_dish=1, Neapolitan=2 }
	feature_values.push_back(2); // meatiness = { vegan=0, meaty=1 }
	feature_values.push_back(4); // presentation = { dine_in=0, take_out=1, delivery=2, frozen=3 }

	// Define the label attributes (or columns)
	std::vector<size_t> label_values;
	label_values.push_back(2); // taste = { lousy=0, delicious=1 }
	label_values.push_back(0); // cost = continuous

	// Make some contrived hard-coded training data
	GClasses::GMatrix features(feature_values);
	GClasses::GMatrix labels(label_values);
	GClasses::GVec f;
	GClasses::GVec l;
	//                     diameter     crust     meatiness presentation                   taste     cost
	f = features.newRow(); f[0] = 14.0; f[1] = 1; f[2] = 1; f[3] = 0; l = labels.newRow(); l[0] = 1; l[1] = 22.95;
	f = features.newRow(); f[0] = 12.0; f[1] = 0; f[2] = 0; f[3] = 3; l = labels.newRow(); l[0] = 0; l[1] = 3.29;
	f = features.newRow(); f[0] = 14.0; f[1] = 1; f[2] = 1; f[3] = 2; l = labels.newRow(); l[0] = 1; l[1] = 15.49;
	f = features.newRow(); f[0] = 12.0; f[1] = 2; f[2] = 0; f[3] = 0; l = labels.newRow(); l[0] = 1; l[1] = 16.65;
	f = features.newRow(); f[0] = 18.0; f[1] = 1; f[2] = 1; f[3] = 3; l = labels.newRow(); l[0] = 0; l[1] = 9.99;
	f = features.newRow(); f[0] = 14.0; f[1] = 1; f[2] = 1; f[3] = 0; l = labels.newRow(); l[0] = 1; l[1] = 14.49;
	f = features.newRow(); f[0] = 12.0; f[1] = 2; f[2] = 0; f[3] = 2; l = labels.newRow(); l[0] = 1; l[1] = 19.65;
	f = features.newRow(); f[0] = 14.0; f[1] = 0; f[2] = 1; f[3] = 1; l = labels.newRow(); l[0] = 0; l[1] = 6.99;
	f = features.newRow(); f[0] = 14.0; f[1] = 1; f[2] = 1; f[3] = 2; l = labels.newRow(); l[0] = 1; l[1] = 19.95;
	f = features.newRow(); f[0] = 14.0; f[1] = 2; f[2] = 0; f[3] = 3; l = labels.newRow(); l[0] = 0; l[1] = 12.99;
	f = features.newRow(); f[0] = 16.0; f[1] = 0; f[2] = 1; f[3] = 0; l = labels.newRow(); l[0] = 0; l[1] = 12.20;
	f = features.newRow(); f[0] = 14.0; f[1] = 1; f[2] = 1; f[3] = 1; l = labels.newRow(); l[0] = 1; l[1] = 15.01;

	// Make a test std::vector
	GClasses::GVec test_features(4);
	GClasses::GVec predicted_labels(2);
	std::cout << "This demo trains and tests several supervised learning models using some contrived hard-coded training data to predict the tastiness and cost of a pizza.\n\n";
	test_features[0] = 15.0; test_features[1] = 2; test_features[2] = 0; test_features[3] = 0;
	std::cout << "Predicting labels for a 15 inch pizza with a Neapolitan-style crust, no meat, for dine-in.\n\n";

	// Use several models to make predictions
	std::cout.precision(4);
	local::decision_tree(features, labels, test_features, predicted_labels);
	std::cout << "The decision tree predicts the taste is " << (predicted_labels[0] == 0 ? "lousy" : "delicious") << ", and the cost is $" << predicted_labels[1] << std::endl;

	local::neural_network(features, labels, test_features, predicted_labels);
	std::cout << "The neural network predicts the taste is " << (predicted_labels[0] == 0 ? "lousy" : "delicious") << ", and the cost is $" << predicted_labels[1] << std::endl;

	local::knn(features, labels, test_features, predicted_labels);
	std::cout << "The knn model predicts the taste is " << (predicted_labels[0] == 0 ? "lousy" : "delicious") << ", and the cost is $" << predicted_labels[1] << std::endl;

	local::naivebayes(features, labels, test_features, predicted_labels);
	std::cout << "The naive Bayes model predicts the taste is " << (predicted_labels[0] == 0 ? "lousy" : "delicious") << ", and the cost is $" << predicted_labels[1] << std::endl;

	local::ensemble(features, labels, test_features, predicted_labels);
	std::cout << "Random forest predicts the taste is " << (predicted_labels[0] == 0 ? "lousy" : "delicious") << ", and the cost is $" << predicted_labels[1] << std::endl;
}

}  // namespace my_waffles
