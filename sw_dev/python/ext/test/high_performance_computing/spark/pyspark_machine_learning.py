#!/usr/bin/env python

# REF [site] >> https://spark.apache.org/docs/latest/ml-guide.html

from pyspark.sql import SparkSession
import pyspark.sql.types as types
import pyspark.sql.functions as func
import pyspark.mllib.stat as mllib_stat
import pyspark.mllib.linalg as mllib_linalg
import pyspark.mllib.feature as mllib_feature
import pyspark.mllib.regression as mllib_regression
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.tree import RandomForest
import pyspark.mllib.evaluation as mllib_eval
import pyspark.ml.feature as ml_feature
import pyspark.ml.classification as ml_classification
import pyspark.ml.evaluation as ml_eval
from pyspark.ml import Pipeline, PipelineModel
import numpy as np
import traceback, sys

# REF [site] >> https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#module-pyspark.ml.classification
def classification_ml():
	if False:
		spark = SparkSession.builder.appName('classification-ml') \
			.config('spark.jars.packages', 'org.xerial:sqlite-jdbc:3.23.1') \
			.getOrCreate()

		df = spark.read \
			.format('jdbc') \
			.option('url', 'jdbc:sqlite:iris.db') \
			.option('driver', 'org.sqlite.JDBC') \
			.option('dbtable', 'iris') \
			.load()
	else:
		spark = SparkSession.builder.appName('classification-ml').getOrCreate()
		df = spark.read.option('header', 'true').option('inferSchema', 'true').format('csv').load('dataset/iris.csv')
	spark.sparkContext.setLogLevel('WARN')
	df.show()

	labels = [
		('index', types.IntegerType()),
		('a1', types.FloatType()),
		('a2', types.FloatType()),
		('a3', types.FloatType()),
		('a4', types.FloatType()),
		('id', types.StringType()),
		('label', types.StringType())
	]

	stringIndexer = ml_feature.StringIndexer(inputCol='label', outputCol='label_int')
	featuresCreator = ml_feature.VectorAssembler(inputCols=[col[0] for col in labels[1:5]], outputCol='features')

	# Create a model.
	logistic = ml_classification.LogisticRegression(featuresCol=featuresCreator.getOutputCol(), labelCol=stringIndexer.getOutputCol(), maxIter=10, regParam=0.01)

	# Create a pipeline.
	pipeline = Pipeline(stages=[stringIndexer, featuresCreator, logistic])

	# Split the dataset into training and testing datasets.
	df_train, df_test = df.randomSplit([0.7, 0.3], seed=666)

	# Run the pipeline and estimate the model.
	model = pipeline.fit(df_train)
	test_result = model.transform(df_test)  # Dataframe.

	#print(test_result.take(1))
	#test_result.show(5, truncate=True, vertical=False)
	test_result.show(truncate=False)

	# Save and load.
	lr_path = './lr'
	logistic.write().overwrite().save(lr_path)
	lr2 = ml_classification.LogisticRegression.load(lr_path)
	print('Param =', lr2.getRegParam())

	model_path = './lr_model'
	model.write().overwrite().save(model_path)
	model2 = PipelineModel.load(model_path)
	print('Stages =', model.stages)
	print(model.stages[2].coefficientMatrix == model2.stages[2].coefficientMatrix)
	print(model.stages[2].interceptVector == model2.stages[2].interceptVector)

# REF [site] >> https://github.com/drabastomek/learningPySpark/blob/master/Chapter05/LearningPySpark_Chapter05.ipynb
def infant_survival_mllib():
	spark = SparkSession.builder.appName('infant-survival-mllib').getOrCreate()
	spark.sparkContext.setLogLevel('WARN')

	labels = [
		('INFANT_ALIVE_AT_REPORT', types.StringType()),
		('BIRTH_YEAR', types.IntegerType()),
		('BIRTH_MONTH', types.IntegerType()),
		('BIRTH_PLACE', types.StringType()),
		('MOTHER_AGE_YEARS', types.IntegerType()),
		('MOTHER_RACE_6CODE', types.StringType()),
		('MOTHER_EDUCATION', types.StringType()),
		('FATHER_COMBINED_AGE', types.IntegerType()),
		('FATHER_EDUCATION', types.StringType()),
		('MONTH_PRECARE_RECODE', types.StringType()),
		('CIG_BEFORE', types.IntegerType()),
		('CIG_1_TRI', types.IntegerType()),
		('CIG_2_TRI', types.IntegerType()),
		('CIG_3_TRI', types.IntegerType()),
		('MOTHER_HEIGHT_IN', types.IntegerType()),
		('MOTHER_BMI_RECODE', types.IntegerType()),
		('MOTHER_PRE_WEIGHT', types.IntegerType()),
		('MOTHER_DELIVERY_WEIGHT', types.IntegerType()),
		('MOTHER_WEIGHT_GAIN', types.IntegerType()),
		('DIABETES_PRE', types.StringType()),
		('DIABETES_GEST', types.StringType()),
		('HYP_TENS_PRE', types.StringType()),
		('HYP_TENS_GEST', types.StringType()),
		('PREV_BIRTH_PRETERM', types.StringType()),
		('NO_RISK', types.StringType()),
		('NO_INFECTIONS_REPORTED', types.StringType()),
		('LABOR_IND', types.StringType()),
		('LABOR_AUGM', types.StringType()),
		('STEROIDS', types.StringType()),
		('ANTIBIOTICS', types.StringType()),
		('ANESTHESIA', types.StringType()),
		('DELIV_METHOD_RECODE_COMB', types.StringType()),
		('ATTENDANT_BIRTH', types.StringType()),
		('APGAR_5', types.IntegerType()),
		('APGAR_5_RECODE', types.StringType()),
		('APGAR_10', types.IntegerType()),
		('APGAR_10_RECODE', types.StringType()),
		('INFANT_SEX', types.StringType()),
		('OBSTETRIC_GESTATION_WEEKS', types.IntegerType()),
		('INFANT_WEIGHT_GRAMS', types.IntegerType()),
		('INFANT_ASSIST_VENTI', types.StringType()),
		('INFANT_ASSIST_VENTI_6HRS', types.StringType()),
		('INFANT_NICU_ADMISSION', types.StringType()),
		('INFANT_SURFACANT', types.StringType()),
		('INFANT_ANTIBIOTICS', types.StringType()),
		('INFANT_SEIZURES', types.StringType()),
		('INFANT_NO_ABNORMALITIES', types.StringType()),
		('INFANT_ANCEPHALY', types.StringType()),
		('INFANT_MENINGOMYELOCELE', types.StringType()),
		('INFANT_LIMB_REDUCTION', types.StringType()),
		('INFANT_DOWN_SYNDROME', types.StringType()),
		('INFANT_SUSPECTED_CHROMOSOMAL_DISORDER', types.StringType()),
		('INFANT_NO_CONGENITAL_ANOMALIES_CHECKED', types.StringType()),
		('INFANT_BREASTFED', types.StringType())
	]
	schema = types.StructType([types.StructField(e[0], e[1], False) for e in labels])
	births = spark.read.csv('dataset/births_train.csv.gz', header=True, schema=schema)

	selected_features = [
		'INFANT_ALIVE_AT_REPORT', 
		'BIRTH_PLACE', 
		'MOTHER_AGE_YEARS', 
		'FATHER_COMBINED_AGE', 
		'CIG_BEFORE', 
		'CIG_1_TRI', 
		'CIG_2_TRI', 
		'CIG_3_TRI', 
		'MOTHER_HEIGHT_IN', 
		'MOTHER_PRE_WEIGHT', 
		'MOTHER_DELIVERY_WEIGHT', 
		'MOTHER_WEIGHT_GAIN', 
		'DIABETES_PRE', 
		'DIABETES_GEST', 
		'HYP_TENS_PRE', 
		'HYP_TENS_GEST', 
		'PREV_BIRTH_PRETERM'
	]
	births_trimmed = births.select(selected_features)

	recode_dictionary = {'YNU': {'Y': 1, 'N': 0, 'U': 0}}  # Yes/No/Unknown.

	def recode(col, key):
		return recode_dictionary[key][col]

	def correct_cig(feat):
		return func.when(func.col(feat) != 99, func.col(feat)).otherwise(0)

	rec_integer = func.udf(recode, types.IntegerType())

	births_transformed = births_trimmed \
		.withColumn('CIG_BEFORE', correct_cig('CIG_BEFORE')) \
		.withColumn('CIG_1_TRI', correct_cig('CIG_1_TRI')) \
		.withColumn('CIG_2_TRI', correct_cig('CIG_2_TRI')) \
		.withColumn('CIG_3_TRI', correct_cig('CIG_3_TRI'))

	cols = [(col.name, col.dataType) for col in births_trimmed.schema]
	YNU_cols = []
	for i, s in enumerate(cols):
		if s[1] == types.StringType():
			dis = births.select(s[0]).distinct().rdd.map(lambda row: row[0]).collect()
			if 'Y' in dis:
				YNU_cols.append(s[0])

	births.select(['INFANT_NICU_ADMISSION', 
		rec_integer('INFANT_NICU_ADMISSION', func.lit('YNU')).alias('INFANT_NICU_ADMISSION_RECODE')
	]).take(5)

	exprs_YNU = [rec_integer(x, func.lit('YNU')).alias(x) if x in YNU_cols else x for x in births_transformed.columns]
	births_transformed = births_transformed.select(exprs_YNU)
	births_transformed.select(YNU_cols[-5:]).show(5)

	# Calculate the descriptive statistics of the numeric features.
	numeric_cols = ['MOTHER_AGE_YEARS','FATHER_COMBINED_AGE',
		'CIG_BEFORE','CIG_1_TRI','CIG_2_TRI','CIG_3_TRI',
		'MOTHER_HEIGHT_IN','MOTHER_PRE_WEIGHT',
		'MOTHER_DELIVERY_WEIGHT','MOTHER_WEIGHT_GAIN'
	]
	numeric_rdd = births_transformed.select(numeric_cols).rdd.map(lambda row: [e for e in row])

	mllib_stats = mllib_stat.Statistics.colStats(numeric_rdd)

	for col, m, v in zip(numeric_cols,  mllib_stats.mean(), mllib_stats.variance()):
		print('{0}: \t{1:.2f} \t {2:.2f}'.format(col, m, np.sqrt(v)))

	# Calculate frequencies for the categorical variables.
	categorical_cols = [e for e in births_transformed.columns if e not in numeric_cols]
	categorical_rdd = births_transformed.select(categorical_cols).rdd.map(lambda row: [e for e in row])

	for i, col in enumerate(categorical_cols):
		agg = categorical_rdd.groupBy(lambda row: row[i]).map(lambda row: (row[0], len(row[1])))
		print(col, sorted(agg.collect(), key=lambda el: el[1], reverse=True))

	# Correlation.
	corrs = mllib_stat.Statistics.corr(numeric_rdd)

	for i, el in enumerate(corrs > 0.5):
		correlated = [(numeric_cols[j], corrs[i][j]) for j, e in enumerate(el) if e == 1.0 and j != i]
		if len(correlated) > 0:
			for e in correlated:
				print('{0}-to-{1}: {2:.2f}'.format(numeric_cols[i], e[0], e[1]))

	# Drop most of highly correlated features.
	features_to_keep = [
		'INFANT_ALIVE_AT_REPORT', 
		'BIRTH_PLACE', 
		'MOTHER_AGE_YEARS', 
		'FATHER_COMBINED_AGE', 
		'CIG_1_TRI', 
		'MOTHER_HEIGHT_IN', 
		'MOTHER_PRE_WEIGHT', 
		'DIABETES_PRE', 
		'DIABETES_GEST', 
		'HYP_TENS_PRE', 
		'HYP_TENS_GEST', 
		'PREV_BIRTH_PRETERM'
	]
	births_transformed = births_transformed.select([e for e in features_to_keep])

	#--------------------
	# Statistical testing.

	# Run a Chi-square test to determine if there are significant differences for categorical variables.
	for cat in categorical_cols[1:]:
	    agg = births_transformed.groupby('INFANT_ALIVE_AT_REPORT').pivot(cat).count()
	    agg_rdd = agg.rdd.map(lambda row: (row[1:])).flatMap(lambda row: [0 if e == None else e for e in row]).collect()

	    row_length = len(agg.collect()[0]) - 1
	    agg = mllib_linalg.Matrices.dense(row_length, 2, agg_rdd)

	    test = mllib_stat.Statistics.chiSqTest(agg)
	    print(cat, round(test.pValue, 4))

	#--------------------
	# Machine learning.

	# Create an RDD of LabeledPoints.
	hashing = mllib_feature.HashingTF(7)

	births_hashed = births_transformed \
		.rdd \
		.map(lambda row: [list(hashing.transform(row[1]).toArray()) if col == 'BIRTH_PLACE' else row[i] for i, col in enumerate(features_to_keep)]) \
		.map(lambda row: [[e] if type(e) == int else e for e in row]) \
		.map(lambda row: [item for sublist in row for item in sublist]) \
		.map(lambda row: mllib_regression.LabeledPoint(row[0], mllib_linalg.Vectors.dense(row[1:])))

	# Split into training and testing.
	births_train, births_test = births_hashed.randomSplit([0.6, 0.4])

	# Estimate a logistic regression model using a stochastic gradient descent (SGD) algorithm.
	LR_Model = LogisticRegressionWithLBFGS.train(births_train, iterations=10)

	# Predict the classes for our testing set.
	LR_results = (
		births_test.map(lambda row: row.label).zip(LR_Model.predict(births_test.map(lambda row: row.features)))
	).map(lambda row: (row[0], row[1] * 1.0))

	# Check how well or how bad our model performed.
	print('********************************************000')
	LR_evaluation = mllib_eval.BinaryClassificationMetrics(LR_results)
	print('********************************************001')
	print('Area under PR: {0:.2f}'.format(LR_evaluation.areaUnderPR))
	print('********************************************002')
	print('Area under ROC: {0:.2f}'.format(LR_evaluation.areaUnderROC))
	print('********************************************003')
	LR_evaluation.unpersist()

	# Select the most predictable features using a Chi-Square selector.
	selector = mllib_feature.ChiSqSelector(4).fit(births_train)

	topFeatures_train = (
		births_train.map(lambda row: row.label).zip(selector.transform(births_train.map(lambda row: row.features)))
	).map(lambda row: mllib_regression.LabeledPoint(row[0], row[1]))

	topFeatures_test = (
		births_test.map(lambda row: row.label).zip(selector.transform(births_test.map(lambda row: row.features)))
	).map(lambda row: mllib_regression.LabeledPoint(row[0], row[1]))

	# Build a random forest model.
	RF_model = RandomForest.trainClassifier(data=topFeatures_train, numClasses=2, categoricalFeaturesInfo={}, numTrees=6, featureSubsetStrategy='all', seed=666)

	RF_results = (topFeatures_test.map(lambda row: row.label).zip(RF_model.predict(topFeatures_test.map(lambda row: row.features))))

	RF_evaluation = mllib_eval.BinaryClassificationMetrics(RF_results)

	print('Area under PR: {0:.2f}'.format(RF_evaluation.areaUnderPR))
	print('Area under ROC: {0:.2f}'.format(RF_evaluation.areaUnderROC))
	RF_evaluation.unpersist()

	# See how the logistic regression would perform with reduced number of features.
	LR_Model_2 = LogisticRegressionWithLBFGS.train(topFeatures_train, iterations=10)

	LR_results_2 = (
		topFeatures_test.map(lambda row: row.label).zip(LR_Model_2.predict(topFeatures_test.map(lambda row: row.features)))
	).map(lambda row: (row[0], row[1] * 1.0))

	LR_evaluation_2 = mllib_eval.BinaryClassificationMetrics(LR_results_2)

	print('Area under PR: {0:.2f}'.format(LR_evaluation_2.areaUnderPR))
	print('Area under ROC: {0:.2f}'.format(LR_evaluation_2.areaUnderROC))
	LR_evaluation_2.unpersist()

def infant_survival_ml():
	spark = SparkSession.builder.appName('infant-survival-ml').getOrCreate()
	spark.sparkContext.setLogLevel('WARN')

	labels = [
		('INFANT_ALIVE_AT_REPORT', types.IntegerType()),
		('BIRTH_PLACE', types.StringType()),
		('MOTHER_AGE_YEARS', types.IntegerType()),
		('FATHER_COMBINED_AGE', types.IntegerType()),
		('CIG_BEFORE', types.IntegerType()),
		('CIG_1_TRI', types.IntegerType()),
		('CIG_2_TRI', types.IntegerType()),
		('CIG_3_TRI', types.IntegerType()),
		('MOTHER_HEIGHT_IN', types.IntegerType()),
		('MOTHER_PRE_WEIGHT', types.IntegerType()),
		('MOTHER_DELIVERY_WEIGHT', types.IntegerType()),
		('MOTHER_WEIGHT_GAIN', types.IntegerType()),
		('DIABETES_PRE', types.IntegerType()),
		('DIABETES_GEST', types.IntegerType()),
		('HYP_TENS_PRE', types.IntegerType()),
		('HYP_TENS_GEST', types.IntegerType()),
		('PREV_BIRTH_PRETERM', types.IntegerType())
	]
	schema = types.StructType([types.StructField(e[0], e[1], False) for e in labels])
	births = spark.read.csv('dataset/births_transformed.csv.gz', header=True, schema=schema)

	# Create transformers.
	births = births.withColumn('BIRTH_PLACE_INT', births['BIRTH_PLACE'].cast(types.IntegerType()))
	# Encode the BIRTH_PLACE column using the OneHotEncoder method.
	encoder = ml_feature.OneHotEncoder(inputCol='BIRTH_PLACE_INT', outputCol='BIRTH_PLACE_VEC')

	featuresCreator = ml_ft.VectorAssembler(inputCols=[col[0] for col in labels[2:]] + [encoder.getOutputCol()], outputCol='features')

	# Create a model.
	logistic = ml_classification.LogisticRegression(maxIter=10, regParam=0.01, labelCol='INFANT_ALIVE_AT_REPORT')

	# Create a pipeline.
	pipeline = Pipeline(stages=[encoder, featuresCreator, logistic])

	# Split the dataset into training and testing datasets.
	births_train, births_test = births.randomSplit([0.7, 0.3], seed=666)

	# Run the pipeline and estimate the model.
	model = pipeline.fit(births_train)
	test_model = model.transform(births_test)

	print(test_model.take(1))

	# Evaluate the performance of the model.
	evaluator = ml_eval.BinaryClassificationEvaluator(rawPredictionCol='probability', labelCol='INFANT_ALIVE_AT_REPORT')
	print(evaluator.evaluate(test_model, {evaluator.metricName: 'areaUnderROC'}))
	print(evaluator.evaluate(test_model, {evaluator.metricName: 'areaUnderPR'}))

	# Save the Pipeline definition.
	pipelinePath = './infant_oneHotEncoder_Logistic_Pipeline'
	pipeline.write().overwrite().save(pipelinePath)

	# Load the Pipeline definition.
	loadedPipeline = Pipeline.load(pipelinePath)
	loadedPipeline.fit(births_train).transform(births_test).take(1)

	# Save the PipelineModel.
	modelPath = './infant_oneHotEncoder_Logistic_PipelineModel'
	model.write().overwrite().save(modelPath)

	# Load the PipelineModel.
	loadedPipelineModel = PipelineModel.load(modelPath)
	test_reloadedModel = loadedPipelineModel.transform(births_test)

	print(test_reloadedModel.take(1))

import pyspark.ml.tuning as tune

def train_validation_splitting_ml():
	spark = SparkSession.builder.appName('train-validation-splitting-ml').getOrCreate()
	spark.sparkContext.setLogLevel('WARN')

	labels = [
		('INFANT_ALIVE_AT_REPORT', types.IntegerType()),
		('BIRTH_PLACE', types.StringType()),
		('MOTHER_AGE_YEARS', types.IntegerType()),
		('FATHER_COMBINED_AGE', types.IntegerType()),
		('CIG_BEFORE', types.IntegerType()),
		('CIG_1_TRI', types.IntegerType()),
		('CIG_2_TRI', types.IntegerType()),
		('CIG_3_TRI', types.IntegerType()),
		('MOTHER_HEIGHT_IN', types.IntegerType()),
		('MOTHER_PRE_WEIGHT', types.IntegerType()),
		('MOTHER_DELIVERY_WEIGHT', types.IntegerType()),
		('MOTHER_WEIGHT_GAIN', types.IntegerType()),
		('DIABETES_PRE', types.IntegerType()),
		('DIABETES_GEST', types.IntegerType()),
		('HYP_TENS_PRE', types.IntegerType()),
		('HYP_TENS_GEST', types.IntegerType()),
		('PREV_BIRTH_PRETERM', types.IntegerType())
	]
	schema = types.StructType([types.StructField(e[0], e[1], False) for e in labels])
	births = spark.read.csv('dataset/births_transformed.csv.gz', header=True, schema=schema)

	# Create transformers.
	births = births.withColumn('BIRTH_PLACE_INT', births['BIRTH_PLACE'].cast(types.IntegerType()))
	# Encode the BIRTH_PLACE column using the OneHotEncoder method.
	encoder = ml_feature.OneHotEncoder(inputCol='BIRTH_PLACE_INT', outputCol='BIRTH_PLACE_VEC')

	featuresCreator = ml_feature.VectorAssembler(inputCols=[col[0] for col in labels[2:]] + [encoder.getOutputCol()], outputCol='features')

	# Split the dataset into training and testing datasets.
	births_train, births_test = births.randomSplit([0.7, 0.3], seed=666)

	# Select only the top five features.
	selector = ml_feature.ChiSqSelector(
		numTopFeatures=5,
		featuresCol=featuresCreator.getOutputCol(),
		outputCol='selectedFeatures',
		labelCol='INFANT_ALIVE_AT_REPORT'
	)

	# Create a purely transforming Pipeline.
	pipeline = Pipeline(stages=[encoder, featuresCreator, selector])
	data_transformer = pipeline.fit(births_train)

	# Create LogisticRegression and Pipeline.
	logistic = ml_classification.LogisticRegression(labelCol='INFANT_ALIVE_AT_REPORT', featuresCol='selectedFeatures')
	grid = tune.ParamGridBuilder() \
		.addGrid(logistic.maxIter, [2, 10, 50]) \
		.addGrid(logistic.regParam, [0.01, 0.05, 0.3]) \
		.build()
	# Define a way of comparing the models.
	evaluator = ml_eval.BinaryClassificationEvaluator(rawPredictionCol='probability', labelCol='INFANT_ALIVE_AT_REPORT')

	# Create a TrainValidationSplit object.
	tvs = tune.TrainValidationSplit(estimator=logistic, estimatorParamMaps=grid, evaluator=evaluator)

	# Fit our data to the model.
	tvsModel = tvs.fit(data_transformer.transform(births_train))
	data_train = data_transformer.transform(births_test)

	# Calculate results.
	results = tvsModel.transform(data_train)
	print(evaluator.evaluate(results, {evaluator.metricName: 'areaUnderROC'}))
	print(evaluator.evaluate(results, {evaluator.metricName: 'areaUnderPR'}))

def hyper_parameter_optimization_ml():
	spark = SparkSession.builder.appName('hyper-parameter-optimization-ml').getOrCreate()
	spark.sparkContext.setLogLevel('WARN')

	labels = [
		('INFANT_ALIVE_AT_REPORT', types.IntegerType()),
		('BIRTH_PLACE', types.StringType()),
		('MOTHER_AGE_YEARS', types.IntegerType()),
		('FATHER_COMBINED_AGE', types.IntegerType()),
		('CIG_BEFORE', types.IntegerType()),
		('CIG_1_TRI', types.IntegerType()),
		('CIG_2_TRI', types.IntegerType()),
		('CIG_3_TRI', types.IntegerType()),
		('MOTHER_HEIGHT_IN', types.IntegerType()),
		('MOTHER_PRE_WEIGHT', types.IntegerType()),
		('MOTHER_DELIVERY_WEIGHT', types.IntegerType()),
		('MOTHER_WEIGHT_GAIN', types.IntegerType()),
		('DIABETES_PRE', types.IntegerType()),
		('DIABETES_GEST', types.IntegerType()),
		('HYP_TENS_PRE', types.IntegerType()),
		('HYP_TENS_GEST', types.IntegerType()),
		('PREV_BIRTH_PRETERM', types.IntegerType())
	]
	schema = types.StructType([types.StructField(e[0], e[1], False) for e in labels])
	births = spark.read.csv('dataset/births_transformed.csv.gz', header=True, schema=schema)

	# Create transformers.
	births = births.withColumn('BIRTH_PLACE_INT', births['BIRTH_PLACE'].cast(types.IntegerType()))
	# Encode the BIRTH_PLACE column using the OneHotEncoder method.
	encoder = ml_feature.OneHotEncoder(inputCol='BIRTH_PLACE_INT', outputCol='BIRTH_PLACE_VEC')

	featuresCreator = ml_feature.VectorAssembler(inputCols=[col[0] for col in labels[2:]] + [encoder.getOutputCol()], outputCol='features')

	# Split the dataset into training and testing datasets.
	births_train, births_test = births.randomSplit([0.7, 0.3], seed=666)

	# Create a purely transforming Pipeline.
	pipeline = Pipeline(stages=[encoder, featuresCreator])
	data_transformer = pipeline.fit(births_train)

	# Specify our model and the list of parameters we want to loop through.
	logistic = ml_classification.LogisticRegression(labelCol='INFANT_ALIVE_AT_REPORT')
	grid = tune.ParamGridBuilder() \
		.addGrid(logistic.maxIter, [2, 10, 50]) \
		.addGrid(logistic.regParam, [0.01, 0.05, 0.3]) \
		.build()
	# Define a way of comparing the models.
	evaluator = ml_eval.BinaryClassificationEvaluator(rawPredictionCol='probability', labelCol='INFANT_ALIVE_AT_REPORT')

	# Create a logic that will do the validation work.
	cv = tune.CrossValidator(estimator=logistic, estimatorParamMaps=grid, evaluator=evaluator)

	cvModel = cv.fit(data_transformer.transform(births_train))

	# See if cvModel performed better than our previous model
	data_train = data_transformer.transform(births_test)
	results = cvModel.transform(data_train)

	print(evaluator.evaluate(results, {evaluator.metricName: 'areaUnderROC'}))
	print(evaluator.evaluate(results, {evaluator.metricName: 'areaUnderPR'}))

	# Parameters which the best model has.
	results = [
		([{key.name: paramValue} for key, paramValue in zip(params.keys(), params.values())], metric)
		for params, metric in zip(cvModel.getEstimatorParamMaps(), cvModel.avgMetrics)
	]
	print(sorted(results, key=lambda el: el[1], reverse=True)[0])

def main():
	classification_ml()

	#infant_survival_mllib()
	#infant_survival_ml()

	#train_validation_splitting_ml()
	#hyper_parameter_optimization_ml()

#%%------------------------------------------------------------------

# Usage:
#	python pyspark_machine_learning.py
#	spark-submit pyspark_machine_learning.py
#	spark-submit --master local[4] pyspark_machine_learning.py
#	spark-submit --master spark://host:7077 --executor-memory 10g pyspark_machine_learning.py

if '__main__' == __name__:
	try:
		main()
	except:
		#ex = sys.exc_info()  # (type, exception object, traceback).
		##print('{} raised: {}.'.format(ex[0], ex[1]))
		#print('{} raised: {}.'.format(ex[0].__name__, ex[1]))
		#traceback.print_tb(ex[2], limit=None, file=sys.stdout)
		#traceback.print_exception(*sys.exc_info(), limit=None, file=sys.stdout)
		traceback.print_exc(limit=None, file=sys.stdout)
