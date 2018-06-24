# REF [site] >>
#	http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
#	http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
#	http://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html

from sklearn import metrics
from sklearn import svm, datasets
from sklearn import model_selection
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
#from scipy import interp
import numpy as np
import matplotlib.pyplot as plt

# REF [site] >>
#	http://scikit-learn.org/stable/modules/model_evaluation.html
#	http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
# For the binary classification task or multilabel classification task in label indicator format.
def classification_model_evaluation_example():
	iris = datasets.load_iris()
	X_train, X_test, y_train, y_test = model_selection.train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)

	clf = svm.SVC(probability=True)
	clf.fit(X_train, y_train)

	y_true = y_test
	if hasattr(clf, 'predict'):
		y_pred = clf.predict(X_test)
	else:
		y_pred = None
	if hasattr(clf, 'predict_proba'):
		y_pred_prob = clf.predict_proba(X_test)
	else:
		y_pred_prob = None
	if hasattr(clf, 'decision_function'):
		y_score = clf.decision_function(X_test)
	elif hasattr(clf, 'predict_proba'):
		y_score = clf.predict_proba(X_test)
	else:
		y_score = None

	print('Classes = {}, {}'.format(np.unique(y_true), np.unique(y_pred)))

	print('Accuracy = {}, {}'.format(metrics.accuracy_score(y_true, y_pred, normalize=True), metrics.accuracy_score(y_true, y_pred, normalize=False)))
	# Logistic loss or cross-entropy loss.
	if y_pred_prob is not None:
		print('Negative log loss = {}, {}'.format(metrics.log_loss(y_true, y_pred_prob, normalize=True), metrics.log_loss(y_true, y_pred_prob, normalize=False)))

	if y_score is not None and 2 == np.unique(y_true).size:
		# For binary classification task or multilabel classification task.
		print('Average precision            =', metrics.average_precision_score(y_true, y_score, average=None))
		print('Average precision (macro)    =', metrics.average_precision_score(y_true, y_score, average='macro'))
		print('Average precision (micro)    =', metrics.average_precision_score(y_true, y_score, average='micro'))
		print('Average precision (weighted) =', metrics.average_precision_score(y_true, y_score, average='weighted'))
		print('Average precision (samples)  =', metrics.average_precision_score(y_true, y_score, average='samples'))

	if y_pred is not None:
		# F1 score, balanced F-score, or F-measure.
		#	F1 = 2 * (precision * recall) / (precision + recall).
		print('F1            =', metrics.f1_score(y_true, y_pred, average=None))
		if 2 == np.unique(y_true).size:
			print('F1 (binary)   =', metrics.f1_score(y_true, y_pred, pos_label=0, average='binary'))
		print('F1 (micro)    =', metrics.f1_score(y_true, y_pred, average='micro'))
		print('F1 (macro)    =', metrics.f1_score(y_true, y_pred, average='macro'))
		print('F1 (weighted) =', metrics.f1_score(y_true, y_pred, average='weighted'))
		# For multilabel classification.
		#print('F1 (samples)  =', metrics.f1_score(y_true, y_pred, average='samples'))

		# Precision = tp / (tp + fp).
		print('Precision            =', metrics.precision_score(y_true, y_pred, average=None))
		if 2 == np.unique(y_true).size:
			print('Precision (binary)   =', metrics.precision_score(y_true, y_pred, pos_label=0, average='binary'))
		print('Precision (micro)    =', metrics.precision_score(y_true, y_pred, average='micro'))
		print('Precision (macro)    =', metrics.precision_score(y_true, y_pred, average='macro'))
		print('Precision (weighted) =', metrics.precision_score(y_true, y_pred, average='weighted'))
		# For multilabel classification.
		#print('Precision (samples)  =', metrics.precision_score(y_true, y_pred, average='samples'))

		# Recall = tp / (tp + fn).
		print('Recall            =', metrics.recall_score(y_true, y_pred, average=None))
		if 2 == np.unique(y_true).size:
			print('Recall (binary)   =', metrics.recall_score(y_true, y_pred, pos_label=0, average='binary'))
		print('Recall (micro)    =', metrics.recall_score(y_true, y_pred, average='micro'))
		print('Recall (macro)    =', metrics.recall_score(y_true, y_pred, average='macro'))
		print('Recall (weighted) =', metrics.recall_score(y_true, y_pred, average='weighted'))
		# For multilabel classification.
		#print('Recall (samples)  =', metrics.recall_score(y_true, y_pred, average='samples'))

	if y_score is not None and 2 == np.unique(y_true).size:
		# Area Under the Receiver Operating Characteristic Curve (ROC AUC).
		# For binary classification task or multilabel classification task.
		print('ROC AUC            =', metrics.roc_auc_score(y_true, y_score, average=None))
		print('ROC AUC (micro)    =', metrics.roc_auc_score(y_true, y_score, average='micro'))
		print('ROC AUC (macro)    =', metrics.roc_auc_score(y_true, y_score, average='macro'))
		print('ROC AUC (weighted) =', metrics.roc_auc_score(y_true, y_score, average='weighted'))
		print('ROC AUC (samples)  =', metrics.roc_auc_score(y_true, y_score, average='samples'))

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
	print('Precision = {}, recall = {}, threshold = {}'.format(precision , recall, thresholds))

	plt.plot(recall, precision)

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
	X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=.5, random_state=0)

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
	classification_model_evaluation_example()

	#confusion_matrix_example()
	#roc_curve_example()
	#precision_recall_curve_example()

	#iris_roc_curve()
	#iris_precision_recall()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
