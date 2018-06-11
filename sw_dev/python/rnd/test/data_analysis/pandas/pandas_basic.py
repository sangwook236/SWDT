import pandas
import numpy as np

#-----------------------------------------------------------

iris_df = pandas.read_csv('./iris.csv', sep=',', header='infer')
print(iris_df.size, iris_df.shape)

# pandas.DataFrame -> np.array.
iris_np = iris_df.values
