#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sklearn.ensemble, sklearn.tree, sklearn.neighbors, sklearn.naive_bayes, sklearn.model_selection, sklearn.metrics, sklearn.datasets

# REF [site] >> http://scikit-learn.org/stable/modules/ensemble.html
def ensemble_example_1():
	X, y =  sklearn.datasets.make_blobs(n_samples=10000, n_features=10, centers=100, random_state=0)
	num_estimators = 10

	clf = sklearn.tree.DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
	scores = sklearn.model_selection.cross_val_score(clf, X, y, cv=5)
	print("DecisionTreeClassifier's score = {}.".format(scores.mean()))

	clf = sklearn.ensemble.RandomForestClassifier(n_estimators=num_estimators, max_depth=None, min_samples_split=2, random_state=0)
	scores = sklearn.model_selection.cross_val_score(clf, X, y, cv=5)
	print("RandomForestClassifier's score = {}.".format(scores.mean()))

	clf = sklearn.ensemble.ExtraTreesClassifier(n_estimators=num_estimators, max_depth=None, min_samples_split=2, random_state=0)
	scores = sklearn.model_selection.cross_val_score(clf, X, y, cv=5)
	print("ExtraTreesClassifier's score = {}.".format(scores.mean()))

	#------------------------------
	X, y = sklearn.datasets.load_iris(return_X_y=True)
	num_estimators = 100

	clf = sklearn.ensemble.BaggingClassifier(sklearn.neighbors.KNeighborsClassifier(), n_estimators=num_estimators, max_samples=0.5, max_features=0.5)
	scores = sklearn.model_selection.cross_val_score(clf, X, y, cv=5)
	print("BaggingClassifier's score = {}.".format(scores.mean()))

	#clf = sklearn.ensemble.AdaBoostClassifier(n_estimators=100)
	clf = sklearn.ensemble.AdaBoostClassifier(sklearn.tree.DecisionTreeClassifier(max_depth=3), n_estimators=num_estimators)
	scores = sklearn.model_selection.cross_val_score(clf, X, y)
	print("AdaBoostClassifier's score = {}.".format(scores.mean()))

	#------------------------------
	X, y = sklearn.datasets.make_hastie_10_2(random_state=0)
	X_train, X_test = X[:2000], X[2000:]
	y_train, y_test = y[:2000], y[2000:]

	clf = sklearn.ensemble.GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train, y_train)
	print("Score = {}.".format(clf.score(X_test, y_test)))
	print("Feature importance = {}.".format(clf.feature_importances_))

	X, y = sklearn.datasets.make_friedman1(n_samples=1200, random_state=0, noise=1.0)
	X_train, X_test = X[:200], X[200:]
	y_train, y_test = y[:200], y[200:]
	est = sklearn.ensemble.GradientBoostingRegressor(
		n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0,
		loss="squared_error"
	).fit(X_train, y_train)
	print("MSE = {}.".format(sklearn.metrics.mean_squared_error(y_test, est.predict(X_test))))

	_ = est.set_params(n_estimators=200, warm_start=True)  # Set warm_start and new nr of trees.
	_ = est.fit(X_train, y_train)  # Fit additional 100 trees to est.
	print("MSE = {}.".format(sklearn.metrics.mean_squared_error(y_test, est.predict(X_test))))

# REF [site] >> http://scikit-learn.org/stable/modules/ensemble.html
def ensemble_example_2():
	iris = sklearn.datasets.load_iris()
	X, y = iris.data[:, 1:3], iris.target

	clf1 = sklearn.linear_model.LogisticRegression(random_state=1)
	clf2 = sklearn.ensemble.RandomForestClassifier(n_estimators=50, random_state=1)
	clf3 = sklearn.naive_bayes.GaussianNB()

	eclf = sklearn.ensemble.VotingClassifier(
		estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],
		voting='hard'
	)

	for clf, label in zip([clf1, clf2, clf3, eclf], ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Ensemble']):
		scores = sklearn.model_selection.cross_val_score(clf, X, y, scoring='accuracy', cv=5)
		print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

	#------------------------------
	# In contrast to majority voting (hard voting), soft voting returns the class label as argmax of the sum of predicted probabilities.

	X, y = iris.data[:, [0, 2]], iris.target

	clf1 = sklearn.tree.DecisionTreeClassifier(max_depth=4)
	clf2 = sklearn.neighbors.KNeighborsClassifier(n_neighbors=7)
	clf3 = sklearn.svm.SVC(kernel='rbf', probability=True)
	eclf = sklearn.ensemble.VotingClassifier(
		estimators=[('dt', clf1), ('knn', clf2), ('svc', clf3)],
		voting='soft', weights=[2, 1, 2]
	)

	clf1 = clf1.fit(X, y)
	clf2 = clf2.fit(X, y)
	clf3 = clf3.fit(X, y)
	eclf = eclf.fit(X, y)

	#------------------------------
	clf1 = sklearn.linear_model.LogisticRegression(random_state=1)
	clf2 = sklearn.ensemble.RandomForestClassifier(random_state=1)
	clf3 = sklearn.naive_bayes.GaussianNB()
	eclf = sklearn.ensemble.VotingClassifier(
		estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],
		voting='soft'
	)

	params = {'lr__C': [1.0, 100.0], 'rf__n_estimators': [20, 200]}

	grid = sklearn.model_selection.GridSearchCV(estimator=eclf, param_grid=params, cv=5)
	grid = grid.fit(iris.data, iris.target)

	#------------------------------
	X, y = sklearn.datasets.load_diabetes(return_X_y=True)

	reg1 = sklearn.ensemble.GradientBoostingRegressor(random_state=1)
	reg2 = sklearn.ensemble.RandomForestRegressor(random_state=1)
	reg3 = sklearn.linear_model.LinearRegression()
	ereg = sklearn.ensemble.VotingRegressor(estimators=[('gb', reg1), ('rf', reg2), ('lr', reg3)])
	ereg = ereg.fit(X, y)

	#------------------------------
	# A stacking predictor predicts as good as the best predictor of the base layer and even sometimes outperforms it by combining the different strengths of the these predictors.
	# However, training a stacking predictor is computationally expensive.

	estimators = [
		('ridge', sklearn.linear_model.RidgeCV()),
		('lasso', sklearn.linear_model.LassoCV(random_state=42)),
		('knr', sklearn.neighbors.KNeighborsRegressor(n_neighbors=20, metric='euclidean'))
	]
	final_estimator = sklearn.ensemble.GradientBoostingRegressor(
		n_estimators=25, subsample=0.5, min_samples_leaf=25, max_features=1,
		random_state=42
	)
	reg = sklearn.ensemble.StackingRegressor(estimators=estimators, final_estimator=final_estimator)

	X, y = sklearn.datasets.load_diabetes(return_X_y=True)

	X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=42)
	reg.fit(X_train, y_train)

	# A StackingRegressor and StackingClassifier can be used as any other regressor or classifier, exposing a predict, predict_proba, and decision_function methods.
	y_pred = reg.predict(X_test)
	print('R2 score: {:.2f}'.format(sklearn.metrics.r2_score(y_test, y_pred)))

	# Get the output of the stacked estimators using the transform method.
	print('Output of the stacked estimators:\n{}'.format(reg.transform(X_test[:5])))

	#------------------------------
	# Multiple stacking layers can be achieved by assigning final_estimator to a StackingClassifier or StackingRegressor.
	final_layer_rfr = sklearn.ensemble.RandomForestRegressor(n_estimators=10, max_features=1, max_leaf_nodes=5,random_state=42)
	final_layer_gbr = sklearn.ensemble.GradientBoostingRegressor(n_estimators=10, max_features=1, max_leaf_nodes=5,random_state=42)
	final_layer = sklearn.ensemble.StackingRegressor(
		estimators=[('rf', final_layer_rfr), ('gbrt', final_layer_gbr)],
		final_estimator=sklearn.linear_model.RidgeCV()
	)
	multi_layer_regressor = sklearn.ensemble.StackingRegressor(
		estimators=[
			('ridge', sklearn.linear_model.RidgeCV()),
			('lasso', sklearn.linear_model.LassoCV(random_state=42)),
			('knr', sklearn.neighbors.KNeighborsRegressor(n_neighbors=20, metric='euclidean'))
		],
		final_estimator=final_layer
	)
	multi_layer_regressor.fit(X_train, y_train)

	print('R2 score: {:.2f}'.format(multi_layer_regressor.score(X_test, y_test)))

def main():
	#ensemble_example_1()
	ensemble_example_2()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
