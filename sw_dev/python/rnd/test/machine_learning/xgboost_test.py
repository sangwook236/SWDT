#!/usr/bin/env python

# REF [site] >> https://github.com/dmlc/xgboost
# REF [site] >> Installation in Windows.
#	https://stackoverflow.com/questions/33749735/how-to-install-xgboost-package-in-python-windows-platform/39811079

import xgboost as xgb
import pandas as pd
import numpy as np
import time

def main():
	# Read in data.
	print('Start preparing data...')
	start_time = time.perf_counter()
	data_filepath = 'D:/work_biz/silicon_minds/datasense_gitlab/engine/test/machine_learning/generated_large_classification_data_1G.csv'
	#data_filepath = 'D:/work_biz/silicon_minds/datasense_gitlab/engine/test/machine_learning/generated_large_classification_data_2G.csv'
	#data_filepath = 'D:/work_biz/silicon_minds/datasense_gitlab/engine/test/machine_learning/generated_large_classification_data_5G.csv'
	#data_filepath = 'D:/work_biz/silicon_minds/datasense_gitlab/engine/test/machine_learning/generated_large_classification_data_10G.csv'
	data_df = pd.read_csv(data_filepath, sep=',', header='infer')
	print('Elapsed time = {}'.format(time.perf_counter() - start_time))
	print('End preparing data...')

	split_idx = int(data_df.shape[0] * 0.75)
	data_dfs = np.split(data_df, [split_idx], axis=0)

	# Specify parameters via map.
	print('Start training XGBoost...')
	start_time = time.perf_counter()
	param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic' }
	num_round = 2
	boost_model = xgb.train(param, data_dfs[0], num_round)
	print('Elapsed time = {}'.format(time.perf_counter() - start_time))
	print('End training XGBoost...')

	# Make prediction.
	print('Start inferring by XGBoost...')
	start_time = time.perf_counter()
	preds = boost_model.predict(data_dfs)
	print('Elapsed time = {}'.format(time.perf_counter() - start_time))
	print('End inferring by XGBoost...')

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
