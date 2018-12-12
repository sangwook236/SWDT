#!/usr/bin/env python

# REF [site] >> https://github.com/IDSIA/sacred

from numpy.random import permutation
from sklearn import svm, datasets
from sacred import Experiment

experiment = Experiment('iris_rbf_svm')

@experiment.config
def cfg():
	C = 1.0
	gamma = 0.7

@experiment.automain
def run(C, gamma):
	iris = datasets.load_iris()
	per = permutation(iris.target.size)
	iris.data = iris.data[per]
	iris.target = iris.target[per]
	clf = svm.SVC(C, 'rbf', gamma=gamma)
	clf.fit(iris.data[:90], iris.target[:90])
	return clf.score(iris.data[90:], iris.target[90:])

# Usage:
#	Mongo Observer:
#		python sacred_basic.py -m SACRED_DB
#		python sacred_basic.py -m HOST:PORT:SACRED_DB
#	File Storage Observer:
#		python sacred_basic.py -F BASEDIR
#		python sacred_basic.py --file_storage=BASEDIR
#	TinyDB Observer:
#		python sacred_basic.py -t BASEDIR
#		python sacred_basic.py --tiny_db=BASEDIR
#	SQL Observer:
#		python sacred_basic.py -s DB_URL
#		python sacred_basic.py --sql=DB_URL
#
# Frontend:
#	Omniboard:
#		omniboard -m HOST:PORT:SACRED_DB
#		omniboard --mu mongodb://user:pwd@host/admin?authMechanism=SCRAM-SHA-1 sacred
#			Open http://localhost:9000 in a web browser.
#	Sacredboard:
#		sacredboard -m SACRED_DB
#		sacredboard -m HOST:PORT:SACRED_DB
#			Open http://127.0.0.1:5000/runs in a web browser.
#		sacredboard -F BASEDIR
