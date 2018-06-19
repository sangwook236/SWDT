# REF [site] >>
#	http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
#	http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
#	http://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html

from sklearn import metrics
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
#from scipy import interp
import numpy as np
import matplotlib.pyplot as plt

def confusion_matrix_example():
	y_true = [2, 0, 2, 2, 0, 1]
	y_pred = [0, 0, 2, 2, 0, 2]
	print('Confusion matrix =', metrics.confusion_matrix(y_true, y_pred))

	y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
	y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
	print('Confusion matrix =', metrics.confusion_matrix(y_true, y_pred, labels=["ant", "bird", "cat"]))

	# In binary classification, the count of true negatives is C_{0,0}, false negatives is C_{1,0}, true positives is C_{1,1} and false positives is C_{0,1}.
	tn, fp, fn, tp = metrics.confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
	print('T/N = {}, F/P = {}, F/N = {}, T/P = {}'.format(tn, fp, fn, tp))

# For the binary classification task.
def roc_curve_example():
	y = np.array([1, 1, 2, 2])
	scores = np.array([0.1, 0.4, 0.35, 0.8])
	fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
	print('False positive rate = {}, true positive rate = {}, threshold = {}'.format(fpr, tpr, thresholds))

	plt.plot(fpr, tpr, 'r-')

	print('AUC  =', metrics.auc(fpr, tpr))

def precision_recall_curve_example():
	y_true = np.array([0, 0, 1, 1])
	y_scores = np.array([0.1, 0.4, 0.35, 0.8])
	precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_scores)
	print('precision = {}, recall = {}, threshold = {}'.format(precision , recall, thresholds))

	plt.plot(recall, precision)

# For the binary classification task or multilabel classification task in label indicator format.
def score_example():
	y_true = np.array([0, 0, 1, 1])
	y_scores = np.array([0.1, 0.4, 0.35, 0.8])

	print('ROC AUC score =', metrics.roc_auc_score(y_true, y_scores))
	print('Average precision score =', metrics.average_precision_score(y_true, y_scores))

# REF [site] >> http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
def iris_roc_curve():
	# Import some data to play with.
	iris = datasets.load_iris()
	X = iris.data
	y = iris.target

	# Binarize the output.
	y = label_binarize(y, classes=[0, 1, 2])
	n_classes = y.shape[1]

	# Add noisy features to make the problem harder.
	random_state = np.random.RandomState(0)
	n_samples, n_features = X.shape
	X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

	# Shuffle and split training and test sets.
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=0)

	# Learn to predict each class against the other.
	classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True, random_state=random_state))
	y_score = classifier.fit(X_train, y_train).decision_function(X_test)

	# Compute ROC curve and ROC area for each class.
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	for i in range(n_classes):
		fpr[i], tpr[i], _ = metrics.roc_curve(y_test[:, i], y_score[:, i])
		roc_auc[i] = metrics.auc(fpr[i], tpr[i])

	# Compute micro-average ROC curve and ROC area.
	fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_test.ravel(), y_score.ravel())
	roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

	plt.figure()
	lw = 2
	plt.plot(fpr[2], tpr[2], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")
	plt.show()

# REF [site] >> http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
def iris_precision_recall():
	iris = datasets.load_iris()
	X = iris.data
	y = iris.target

	# Add noisy features.
	random_state = np.random.RandomState(0)
	n_samples, n_features = X.shape
	X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

	# Limit to the two first classes, and split into training and test.
	X_train, X_test, y_train, y_test = train_test_split(X[y < 2], y[y < 2], test_size=.5, random_state=random_state)

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
	#confusion_matrix_example()
	#roc_curve_example()
	#precision_recall_curve_example()
	#score_example()

	iris_roc_curve()
	#iris_precision_recall()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
