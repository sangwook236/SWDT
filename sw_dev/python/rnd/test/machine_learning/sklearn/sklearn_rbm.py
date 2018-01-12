# REF [site] >> http://scikit-learn.org/stable/modules/neural_networks_unsupervised.html
# REF [site] >> http://scikit-learn.org/stable/auto_examples/neural_networks/plot_rbm_logistic_classification.html#sphx-glr-auto-examples-neural-networks-plot-rbm-logistic-classification-py

import numpy as np
from sklearn.neural_network import BernoulliRBM

X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
model = BernoulliRBM(n_components=2)
model.fit(X)
