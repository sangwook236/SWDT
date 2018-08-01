#!/usr/bin/env python

# REF [site] >>
#	http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
#	http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
#	http://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html

from sklearn import metrics
from sklearn import model_selection, multiclass
from sklearn import svm, ensemble, datasets, preprocessing
#from scipy import interp
import numpy as np
import matplotlib.pyplot as plt

# REF [site] >>
#	http://scikit-learn.org/stable/modules/model_evaluation.html
#	http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
# For binary classification task or multilabel classification task in label indicator format.
def classification_model_evaluation_example(is_binary_classification, classifier):
	if is_binary_classification:
		X, Y = datasets.make_classification(n_samples=200, n_features=4, n_informative=4, n_redundant=0, n_repeated=0, n_classes=2, shuffle=False, random_state=0)
		#Y[Y == 1] = -21
		X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.4, shuffle=True, random_state=0)
		pos_label = 1  # Actual label, not index.
	else:
		if True:
			iris = datasets.load_iris()
			X_train, X_test, y_train, y_test = model_selection.train_test_split(iris.data, iris.target, test_size=0.4, shuffle=True, random_state=0)
			pos_label = 2  # Actual label, not index.
	
			"""
			# After these steps, y_test has only two labels {0, 1}.
			X_test = X_test[2 != y_test]
			y_test = y_test[2 != y_test]
			X_test, y_test = X_test[:25], y_test[:25]
			"""
		else:
			X, Y = datasets.make_classification(n_samples=200, n_features=10, n_informative=10, n_redundant=0, n_repeated=0, n_classes=5, shuffle=False, random_state=0)
			#Y[Y == 1] = -21
			X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.4, shuffle=True, random_state=0)
			pos_label = 2  # Actual label, not index.

	#--------------------
	classifier.fit(X_train, y_train)

	if hasattr(classifier, 'predict'):
		y_pred = classifier.predict(X_test)
	else:
		y_pred = None
	if hasattr(classifier, 'predict_proba'):
		y_pred_prob = classifier.predict_proba(X_test)
	else:
		y_pred_prob = None
	if hasattr(classifier, 'decision_function'):
		# y_score.shape = [num_samples] if binary classification.
		# y_score.shape = [num_samples, num_classes] otherwise.
		y_score = classifier.decision_function(X_test)
	elif hasattr(classifier, 'predict_proba'):
		# y_score.shape = [num_samples, num_classes].
		y_score = classifier.predict_proba(X_test)
	else:
		y_score = None

	#--------------------
	y_true = y_test
	"""
	if y_score is not None:
		num_classes = max(np.unique(y_true).size, np.unique(y_pred).size, y_score.shape[1])
		print('#classes = {} = max({}, {}, {})'.format(num_classes, np.unique(y_true).size, np.unique(y_pred).size, y_score.shape[1]))
	else:
		num_classes = max(np.unique(y_true).size, np.unique(y_pred).size)
		print('#classes = {} = max({}, {})'.format(num_classes, np.unique(y_true).size, np.unique(y_pred).size))
	"""
	# classifier.classes_ saves actual labels [-7, 2, 3, 21, 37], not their indices [0, 1, 2, 3, 4].
	classes = classifier.classes_.tolist()
	num_classes = len(classes)

	print('Classification info:')
	print('\tClassifier =', type(classifier))
	print('\tClass labels = {}, #classes = {}'.format(classes, num_classes))
	print('\tY_true.shape = {}, Y_pred.shape = {}, Y_score.shape = {}'.format(y_true.shape, y_pred.shape if y_pred is not None else None, y_score.shape if y_score is not None else None))

	#--------------------
	# Accuracy.
	print('Accuracy = {}, {}'.format(metrics.accuracy_score(y_true, y_pred, normalize=True), metrics.accuracy_score(y_true, y_pred, normalize=False)))
	# Logistic loss or cross-entropy loss.
	if y_pred_prob is not None:
		print('Negative log loss = {}, {}'.format(metrics.log_loss(y_true, y_pred_prob, normalize=True), metrics.log_loss(y_true, y_pred_prob, normalize=False)))

	# Average precision.
	#	For binary classification task or multilabel classification task.
	if y_score is not None:
		if 2 == num_classes:
			y_true2 = y_true
			if 1 == y_score.ndim:
				y_score2 = y_score
			else:
				pos_idx = classes.index(pos_label)
				y_score2 = y_score[:,pos_idx]
		else:
			pos_indices = pos_label == y_true
			neg_indices = pos_label != y_true
			y_true2 = y_true.copy()
			y_true2[pos_indices] = 1
			y_true2[neg_indices] = 0
			pos_idx = classes.index(pos_label)
			y_score2 = y_score[:,pos_idx]

		print('Average precision            =', metrics.average_precision_score(y_true2, y_score2, average=None))
		print('Average precision (macro)    =', metrics.average_precision_score(y_true2, y_score2, average='macro'))
		print('Average precision (micro)    =', metrics.average_precision_score(y_true2, y_score2, average='micro'))
		print('Average precision (weighted) =', metrics.average_precision_score(y_true2, y_score2, average='weighted'))
		print('Average precision (samples)  =', metrics.average_precision_score(y_true2, y_score2, average='samples'))

	# F1 score, balanced F-score, or F-measure.
	#	F1 = 2 * (precision * recall) / (precision + recall).
	if y_pred is not None:
		print('F1            =', metrics.f1_score(y_true, y_pred, average=None))
		if 2 == num_classes:
			# NOTE [info] >> Whether it is a binary classifcation task or not is decided by the number of classes which y_true & y_pred have.
			print('F1 (binary)   =', metrics.f1_score(y_true, y_pred, pos_label=pos_label, average='binary'))  # pos_label assigns one label to be used as positive out of two labels.
		print('F1 (micro)    =', metrics.f1_score(y_true, y_pred, average='micro'))
		print('F1 (macro)    =', metrics.f1_score(y_true, y_pred, average='macro'))
		print('F1 (weighted) =', metrics.f1_score(y_true, y_pred, average='weighted'))
		# For multilabel classification.
		#print('F1 (samples) =', metrics.f1_score(y_true, y_pred, average='samples'))

	# Precision = tp / (tp + fp).
	if y_pred is not None:
		print('Precision            =', metrics.precision_score(y_true, y_pred, average=None))
		if 2 == num_classes:
			# NOTE [info] >> Whether it is a binary classifcation task or not is decided by the number of classes which y_true & y_pred have.
			print('Precision (binary)   =', metrics.precision_score(y_true, y_pred, pos_label=pos_label, average='binary'))  # pos_label assigns one label to be used as positive out of two labels.
		print('Precision (micro)    =', metrics.precision_score(y_true, y_pred, average='micro'))
		print('Precision (macro)    =', metrics.precision_score(y_true, y_pred, average='macro'))
		print('Precision (weighted) =', metrics.precision_score(y_true, y_pred, average='weighted'))
		# For multilabel classification.
		#print('Precision (samples) =', metrics.precision_score(y_true, y_pred, average='samples'))

	# Recall = tp / (tp + fn).
	if y_pred is not None:
		print('Recall            =', metrics.recall_score(y_true, y_pred, average=None))
		if 2 == num_classes:
			# NOTE [info] >> Whether it is a binary classifcation task or not is decided by the number of classes which y_true & y_pred have.
			print('Recall (binary)   =', metrics.recall_score(y_true, y_pred, pos_label=pos_label, average='binary'))  # pos_label assigns one label to be used as positive out of two labels.
		print('Recall (micro)    =', metrics.recall_score(y_true, y_pred, average='micro'))
		print('Recall (macro)    =', metrics.recall_score(y_true, y_pred, average='macro'))
		print('Recall (weighted) =', metrics.recall_score(y_true, y_pred, average='weighted'))
		# For multilabel classification.
		#print('Recall (samples) =', metrics.recall_score(y_true, y_pred, average='samples'))

	# Area Under the Receiver Operating Characteristic Curve (ROC AUC).
	#	For binary classification task or multilabel classification task.
	if y_score is not None:
		if 2 == num_classes:
			y_true2 = y_true
			if 1 == y_score.ndim:
				y_score2 = y_score
			else:
				pos_idx = classes.index(pos_label)
				y_score2 = y_score[:,pos_idx]
		else:
			pos_indices = pos_label == y_true
			neg_indices = pos_label != y_true
			y_true2 = y_true.copy()
			y_true2[pos_indices] = 1
			y_true2[neg_indices] = 0
			pos_idx = classes.index(pos_label)
			y_score2 = y_score[:,pos_idx]

		print('ROC AUC            =', metrics.roc_auc_score(y_true2, y_score2, average=None))
		print('ROC AUC (micro)    =', metrics.roc_auc_score(y_true2, y_score2, average='micro'))
		print('ROC AUC (macro)    =', metrics.roc_auc_score(y_true2, y_score2, average='macro'))
		print('ROC AUC (weighted) =', metrics.roc_auc_score(y_true2, y_score2, average='weighted'))
		print('ROC AUC (samples)  =', metrics.roc_auc_score(y_true2, y_score2, average='samples'))

	# ROC curve.
	#	For binary classification task.
	if y_score is not None:
		if 1 == y_score.ndim:
			y_score2 = y_score
		else:
			pos_idx = classes.index(pos_label)
			y_score2 = y_score[:,pos_idx]

		# y_score.shape must be [num_samples].
		# If pos_label is assigned, binary classification task is set using one vs. the others.
		fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score2, pos_label=pos_label)
		print('False positive rate = {}\nTrue positive rate = {}\nThreshold = {}'.format(fpr, tpr, thresholds))

	# Precision-Recall curve.
	#	For binary classification task.
	if y_score is not None:
		if 1 == y_score.ndim:
			y_score2 = y_score
		else:
			pos_idx = classes.index(pos_label)
			y_score2 = y_score[:,pos_idx]

		# y_score.shape must be [num_samples].
		# If pos_label is assigned, binary classification task is set using one vs. the others.
		precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_score2, pos_label=pos_label)
		print('Precision = {}\nRecall = {}\nThreshold = {}'.format(precision , recall, thresholds))

def confusion_matrix_example():
	y_true = [2, 0, 2, 2, 0, 1]
	y_pred = [0, 0, 2, 2, 0, 2]
	print('Confusion matrix =', metrics.confusion_matrix(y_true, y_pred))

	y_true = ['cat', 'ant', 'cat', 'cat', 'ant', 'bird']
	y_pred = ['ant', 'ant', 'cat', 'cat', 'ant', 'cat']
	print('Confusion matrix =', metrics.confusion_matrix(y_true, y_pred, labels=['ant', 'bird', 'cat']))

	# In binary classification, the count of true negatives is C_{0,0}, false negatives is C_{1,0}, true positives is C_{1,1} and false positives is C_{0,1}.
	tn, fp, fn, tp = metrics.confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
	print('T/N = {}, F/P = {}, F/N = {}, T/P = {}.'.format(tn, fp, fn, tp))

# For binary classification task.
def roc_curve_example(is_binary_classification):
	if is_binary_classification:
		y = np.array([1, 1, 3, 3])
	else:
		y = np.array([1, 1, 3, 4])
	scores = np.array([0.1, 0.4, 0.35, 0.8])
	fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=3)  # If pos_label is assigned, binary classification task is set using one vs. the others.
	print('False positive rate = {}, true positive rate = {}, threshold = {}'.format(fpr, tpr, thresholds))

	plt.plot(fpr, tpr, 'r-')

	print('AUC  =', metrics.auc(fpr, tpr))

# For binary classification task.
def precision_recall_curve_example(is_binary_classification):
	if is_binary_classification:
		y = np.array([1, 1, 3, 3])
	else:
		y = np.array([1, 1, 3, 4])
	y_scores = np.array([0.1, 0.4, 0.35, 0.8])
	precision, recall, thresholds = metrics.precision_recall_curve(y, y_scores, pos_label=1)  # If pos_label is assigned, binary classification task is set using one vs. the others.
	print('Precision = {}, recall = {}, threshold = {}'.format(precision , recall, thresholds))

	plt.plot(recall, precision)

# For binary classification task..
# REF [site] >> http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
def iris_roc_curve():
	# Import some data to play with.
	iris = datasets.load_iris()
	X = iris.data
	y = iris.target

	# Binarize the output.
	y = preprocessing.label_binarize(y, classes=[0, 1, 2])
	n_classes = y.shape[1]

	# Add noisy features to make the problem harder.
	random_state = np.random.RandomState(0)
	n_samples, n_features = X.shape
	X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

	# Shuffle and split training and test sets.
	X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=.5, random_state=0)

	# Learn to predict each class against the other.
	classifier = multiclass.OneVsRestClassifier(svm.SVC(kernel='linear', probability=True, random_state=random_state))
	y_score = classifier.fit(X_train, y_train).decision_function(X_test)

	# Compute ROC curve and ROC area for each class.
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	for i in range(n_classes):
		fpr[i], tpr[i], _ = metrics.roc_curve(y_test[:, i], y_score[:, i])
		roc_auc[i] = metrics.auc(fpr[i], tpr[i])

	# Compute micro-average ROC curve and ROC area.
	fpr['micro'], tpr['micro'], _ = metrics.roc_curve(y_test.ravel(), y_score.ravel())
	roc_auc['micro'] = metrics.auc(fpr['micro'], tpr['micro'])

	plt.figure()
	lw = 2
	plt.plot(fpr[2], tpr[2], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc='lower right')
	plt.show()

# For binary classification task.
# REF [site] >> http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
def iris_precision_recall_curve():
	iris = datasets.load_iris()
	X = iris.data
	y = iris.target

	# Add noisy features.
	random_state = np.random.RandomState(0)
	n_samples, n_features = X.shape
	X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

	# Limit to the two first classes, and split into training and test.
	X_train, X_test, y_train, y_test = model_selection.train_test_split(X[y < 2], y[y < 2], test_size=.5, random_state=random_state)

	# Create a simple classifier.
	classifier = svm.LinearSVC(random_state=random_state)
	classifier.fit(X_train, y_train)
	y_score = classifier.decision_function(X_test)

	average_precision = metrics.average_precision_score(y_test, y_score)
	print('Average precision-recall score: {0:0.2f}.'.format(average_precision))

	precision, recall, _ = metrics.precision_recall_curve(y_test, y_score)

	plt.step(recall, precision, color='b', alpha=0.2, where='post')
	plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.ylim([0.0, 1.05])
	plt.xlim([0.0, 1.0])
	plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))

def main():
	#classifier = svm.SVC(probability=True)
	classifier = ensemble.RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=2, random_state=0)

	is_binary_classification = True

	#--------------------
	classification_model_evaluation_example(is_binary_classification, classifier)

	#confusion_matrix_example()
	#roc_curve_example(is_binary_classification)
	#precision_recall_curve_example(is_binary_classification)

	#iris_roc_curve()
	#iris_precision_recall_curve()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
