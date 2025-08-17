#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np

#---------------------------------------------------------------------

# REF [site] >> http://pandas.pydata.org/pandas-docs/stable/10min.html
# REF [site] >> http://pandas.pydata.org/pandas-docs/stable/tutorials.html
def basic_operation():
	dates = pd.date_range('20130101', periods=6)
	df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
	print('df =\n', df, sep='')

	if False:
		# Show the data frame.
		from IPython.display import display, HTML
		display(df)
		#print(HTML(df.to_html()).data)

	df2 = pd.DataFrame({
		'A': 1.,
		'B': pd.Timestamp('20130102'),
		'C': pd.Series(1, index=list(range(4)), dtype='float32'),
		'D': np.array([3] * 4, dtype='int32'),
		'E': pd.Categorical(["test", "train", "test", "train"]),
		'F': 'foo'
	})
	print('df =\n', df2, sep='')
	print('df.dtypes =\n', df2.dtypes, sep='')

	print('df.head() =\n', df.head(), sep='')
	print('df.tail(3) =\n', df.tail(3), sep='')

	print('df.index =\n', df.index, sep='')
	print('df.columns =\n', df.columns, sep='')
	print('df.values =\n', df.values, sep='')  # np.ndarray.

	# REF [site] >> https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.describe.html
	print('df.describe() =\n', df.describe(), sep='')

def indexing_and_slicing():
	iris_df = pd.read_csv('./iris.csv', sep=',', header='infer')

	if False:
		# Show the data frame.
		from IPython.display import display, HTML
		display(iris_df)
		#print(HTML(iris_df.to_html()).data)

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

# REF [site] >> https://pandas.pydata.org/pandas-docs/stable/missing_data.html
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

def value_operation():
	pd.options.mode.use_inf_as_na = True

	csv_filepath = './titanic_train.csv'
	df = pd.read_csv(csv_filepath, sep=',', header='infer', index_col='PassengerId')

	#df[df.Sex == 'female'].sum()
	#df[df['Sex'] == 'female'].Survived.cumsum()
	print('#survived females =', df[df.Sex == 'female'].count()['Survived'])

	print('Unique values of Embarked =', df.Embarked.unique())

	# Drop columns.
	df = df.drop(columns=['Name', 'Ticket', 'Cabin'])
	#df = df.drop(columns=['Name'])

	# Replace values.
	df.Sex = df.Sex.replace(['male', 'female'], [0, 1])
	df.Embarked = df.Embarked.replace(['C', 'S', 'Q'], [0, 1, 2])

	# Impute missing values.
	df.Age = df.Age.fillna(0)
	#values = {'Age': 0, 'Embarked': -1, 'Cabin': 'unknown'}
	values = {'Embarked': -1}
	df = df.fillna(value=values)

	#print(df)
	
	# Write to a CSV file.
	df.to_csv('./titanic_train_revised.csv', na_rep='N/A')

def merged_cell_example():
	# REF [site] >> https://stackoverflow.com/questions/46600472/avoid-merged-cells-in-pandas-to-excel-method
	df = pd.DataFrame({
		"animal": ("horse", "horse", "dog", "dog"),
		"color of fur": ("black", "white", "grey", "black"),
		"name": ("Blacky", "Wendy", "Rufus", "Catchy")
	})
	df_merged = df.set_index(["animal", "color of fur"])  # Necessity.

	df_merged.to_excel("./merged_cell.xlsx")  # Merged cells exist.
	df_merged.to_html("./merged_cell.html")  # Merged cells exist.
	df_merged.to_csv("./merged_cell.csv", na_rep="N/A")  # Merged cells do not exist.

	if True:
		from IPython.display import display, HTML
		display(df_merged)
		print(HTML(df_merged.to_html()).data)

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
	# GPU acceleration:
	#	https://developer.nvidia.com/blog/7-drop-in-replacements-to-instantly-speed-up-your-python-data-science-workflows

	if False:
		# Set 4 options to display complete contents of a dataframe without any kind of truncation.
		pd.set_option('display.max_rows', None)
		pd.set_option('display.max_columns', None)
		pd.set_option('display.width', None)
		pd.set_option('display.max_colwidth', None)

	#basic_operation()
	#indexing_and_slicing()
	#function_operation()
	#value_operation()

	merged_cell_example()

	#numpy_operation()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
