#!/usr/bin/env python

import pandas as pd
import numpy as np

#%%-------------------------------------------------------------------

# REF [site] >> http://pandas.pydata.org/pandas-docs/stable/10min.html
def basic_operation():
	dates = pd.date_range('20130101', periods=6)
	df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
	print('df =\n', df, sep='')

	df2 = pd.DataFrame({
		'A' : 1.,
		'B' : pd.Timestamp('20130102'),
		'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
		'D' : np.array([3] * 4,dtype='int32'),
		'E' : pd.Categorical(["test","train","test","train"]),
		'F' : 'foo'
	})
	print('df =\n', df2, sep='')
	print('df.dtypes =\n', df2.dtypes, sep='')

	print('df.head() =\n', df.head(), sep='')
	print('df.tail(3) =\n', df.tail(3), sep='')

	print('df.index =\n', df.index, sep='')
	print('df.columns =\n', df.columns, sep='')
	print('df.values =\n', df.values, sep='')  # np.ndarray.

	print('df.describe() =\n', df.describe(), sep='')

def indexing_and_slicing():
	iris_df = pd.read_csv('./iris.csv', sep=',', header='infer')

	print('Size = {}, shape = {}'.format(iris_df.size, iris_df.shape))

	# Indexing and slicing.
	print('Columns =\n', iris_df[['a1', 'a3']], sep='')
	print('Rows =\n', iris_df[0:10], sep='')
	#print('Submatrix =\n', iris_df[5:20, 'a1':'a3'], sep='')  # Error.

	print('Rows (loc) =\n', iris_df.loc[0:10], sep='')
	print('Submatrix (loc) =\n', iris_df.loc[5:20, 'a1':'a3'], sep='')
	print('Submatrix (loc) =\n', iris_df.loc[5:20, ['a1', 'a3', 'a4']], sep='')
	#print('Submatrix (loc) =\n', iris_df.loc[5:20, 1:3], sep='')  # Error.

	print('Rows (iloc) =\n', iris_df.iloc[0:10], sep='')
	#print('Submatrix (iloc) =\n', iris_df.iloc[5:20, 'a1':'a3'], sep='')  # Error.
	print('Submatrix (iloc) =\n', iris_df.iloc[5:20, 1:3], sep='')
	print('Submatrix (iloc) =\n', iris_df.iloc[5:20, [0, 1, 3]], sep='')

def function_operation():
	#df1 = pd.DataFrame(data=[[1, 7, 3, 2, 5, 2, 5, 3, 4, 7, 9, -1, 2, 5, -1, 7]])  # Row.
	df1 = pd.DataFrame(data=[1, 7, 3, 2, 5, 2, 5, 3, 4, 7, 9, -1, 2, 5, -1, 7])  # Column.
	print('Size = {}, shape = {}, data =\n{}'.format(df1.size, df1.shape, df1), sep='')

	print('pd.unique(df1) =', pd.unique(df1.iloc[:,0]))  # np.ndarray.

	df2 = pd.DataFrame(data=[[1, 7, 3, 2, 5, 2, 5, 3], [4, 7, 9, -1, 2, 5, -1, 7]])
	print('Size = {}, shape = {}, data =\n{}'.format(df2.size, df2.shape, df2), sep='')

	print('pd.unique(df2) =', pd.unique(df2.iloc[:, 0]))
	print('np.sort(pd.unique(df2)) =', np.sort(pd.unique(df2.iloc[0, :])))
	#print('pd.unique(df2) =', pd.unique(df2))  # Error.

def numpy_operation():
	iris_df = pd.read_csv('./iris.csv', sep=',', header='infer')
	print('Data frame =\n', iris_df, sep='')

	# pd.DataFrame -> np.ndarray.
	iris_np = iris_df.values
	print('np.ndarray from pd.DataFrame =\n', iris_np, sep='')

	# np.ndarray -> pd.DataFrame.
	iris_df2 = pd.DataFrame(data=iris_np, columns=['a1', 'a2', 'a3', 'a4', 'id', 'label'])
	#iris_df2 = pd.DataFrame(data=iris_np)
	print('pd.DataFrame from np.ndarray =\n', iris_df2, sep='')

def main():
	basic_operation()
	#indexing_and_slicing()
	#function_operation()

	#numpy_operation()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
