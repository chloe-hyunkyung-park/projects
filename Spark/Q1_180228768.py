#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
COM6012 Scalable Machine Learning
Assignment 02
180228768
Question 1. Searching for exotic particles in high-energy physics 
​using classic supervised learning algorithms ​[10 marks]
'''
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
from pyspark.ml import Transformer
from pyspark.ml.feature import VectorAssembler, Binarizer
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from heapq import nlargest   
import time
import findspark
findspark.init()

#################################################################################################
#HPC - 16CORES
spark = SparkSession.builder\
        .config("spark.locl.dir", "/fastdata/acp18hp")\
        .appName("AS02")\
        .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")

#################################################################################################
# HPC - read data
data = spark.read.option("inferschema",False).csv("/fastdata/acp18hp/HIGGS.csv.gz").repartition(10).cache()
#################################################################################################
# Data preprocessing
col_names = ['label', 'lepton_pT', 'lepton_eta', 'lepton_phi', 'missing_energy_magnitude', 'missing_energy_phi', 
                'jet_1_pt', 'jet_1_eta', 'jet_1_phi', 'jet_1_b_tag', 'jet_2_pt', 'jet_2_eta', 'jet_2_phi', 'jet_2_b_tag', 
                'jet_3_pt', 'jet_3_eta', 'jet_3_phi', 'jet_3_b_tag', 'jet_4_pt', 'jet_4_eta', 'jet_4_phi', 'jet_4_b_tag', 
                'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb']
feat_names = col_names[1:]

# To parse strings into integers
class ChangeColNames(Transformer):  
    def __init__(self, columns=[None]):
        self.columns = columns
    
    def _transform(self, df):
        for i in range(len(self.columns)):
            df = df.withColumnRenamed(self.columns[i], col_names[i])
        self = df
        return self
    
# To parse strings into doubleType
class ParserDouble(Transformer):  
    def __init__(self, columns=[None]):
        self.columns = columns
    
    def _transform(self, df):
        for col in self.columns:
            df = df.withColumn(col, df[col].cast(DoubleType()))
        self = df
        return self
# Before changing the column names, argument should be original columns of data. 
change_col = ChangeColNames(data.columns)
double_maker = ParserDouble(col_names)
assembler = VectorAssembler(inputCols = col_names[1:], outputCol = 'features')
# Pipeline for preprocessing data
pipeline = Pipeline(stages=[change_col, double_maker, assembler])
data = pipeline.fit(data).transform(data).select('features', 'label')
#################################################################################################
# For getting classification AUC 
evaluator1 = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction")
# For getting Accuracy
evaluator2 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
# pipeline for crossVaildator
pipeline = Pipeline(stages = [])

# data split
# sample : for training to get the best model through crossValidator
sample = data.sample(False, 0.25)
# train, test : for training and testing each model 
train, test = sample.randomSplit([0.8, 0.2], seed = 180228768)
#################################################################################################
###DecisionTreeClassifier
#m1 : dtc
dtc = DecisionTreeClassifier(labelCol = 'label', featuresCol = 'features')
dtc_grid = (ParamGridBuilder()
            .baseOn({pipeline.stages: [dtc]}) \
            .addGrid(dtc.maxDepth, [5, 6, 7])
            .addGrid(dtc.maxBins, [5, 15, 30])
            .build())
dtc_cv = CrossValidator(estimator = pipeline,
                    estimatorParamMaps = dtc_grid, 
                    evaluator = evaluator2, 
                    numFolds = 3,
                    parallelism = 10)
dtc_cvModel = dtc_cv.fit(sample)

#Extract the best configuration
best_dtc = dtc_cvModel.bestModel.stages[-1]._java_obj
best_maxDepth = best_dtc.getMaxDepth()
best_maxBins = best_dtc.getMaxBins()

#create dtc with the best configuration
dtc = DecisionTreeClassifier(labelCol = 'label', featuresCol = 'features',
                            maxDepth = best_maxDepth, maxBins = best_maxBins )
dtc_s_time = time.time()
dtc_model = dtc.fit(train)
dtc_e_time = time.time()
dtc_predictions = dtc_model.transform(test)
dtc_auc = evaluator1.evaluate(dtc_predictions)
dtc_accuracy = evaluator2.evaluate(dtc_predictions)
print("AUC of Decision Tree Classifier = %g " % dtc_auc)
print("Accuracy of Decision Tree Classifier = %g " % dtc_accuracy)
dtc_imp_feat = list(dtc_model.featureImportances)
indice = nlargest(3, range(len(dtc_imp_feat)), key=lambda i: dtc_imp_feat[i])
print([feat_names[i] for i in indice])
dtc_time = (dtc_e_time - dtc_s_time)
print("Learning time of dtc = ", dtc_time)
#################################################################################################
###DecisionTreeRegressor
#m2 : dtr
evaluator1 = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="bi_prediction")
evaluator2 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="bi_prediction", metricName="accuracy")

dtr = DecisionTreeRegressor(labelCol="label", featuresCol="features")
binarizer = Binarizer(threshold=0.5, inputCol="prediction", outputCol="bi_prediction")
dtr_grid = (ParamGridBuilder()
            .baseOn({pipeline.stages: [dtr, binarizer]}) \
            .addGrid(dtr.maxBins, [5, 10, 30])
            .addGrid(dtr.maxDepth, [5, 6, 7])
            .build())
dtr_cv = CrossValidator(estimator = pipeline,
                    estimatorParamMaps = dtr_grid, 
                    evaluator = evaluator2, 
                    numFolds = 3,
                    parallelism = 10)
dtr_cvModel = dtr_cv.fit(sample)

#Extract the best configuration
best_dtr = dtr_cvModel.bestModel.stages[0]._java_obj
dtr_maxDepth = best_dtr.getMaxDepth()
dtr_maxBins = best_dtr.getMaxBins()

#create dtr with the best configuration
dtr = DecisionTreeRegressor(labelCol = 'label', featuresCol = 'features',
                            maxDepth = dtr_maxDepth, maxBins = dtr_maxBins )
dtr_s_time = time.time()
dtr_model = dtr.fit(train)
dtr_e_time = time.time()

#binarizer
dtr_predictions = dtr_model.transform(test)
dtr_predictions = binarizer.transform(dtr_predictions)

dtr_auc = evaluator1.evaluate(dtr_predictions)
dtr_accuracy = evaluator2.evaluate(dtr_predictions)
print("AUC of DecisionTreeRegressor = %g " % dtr_auc)
print("Accuracy of DecisionTreeRegressor = %g " % dtr_accuracy)
dtr_imp_feat = list(dtr_model.featureImportances)
indice = nlargest(3, range(len(dtr_imp_feat)), key=lambda i: dtr_imp_feat[i])
print([feat_names[i] for i in indice])
dtr_time = (dtr_e_time - dtr_s_time) 
print("Learning time of dtr = ", dtr_time)
#################################################################################################
###LogisticRegression
#m3 : LogisticRegression,lr
evaluator1 = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction")
evaluator2 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

lr = LogisticRegression(labelCol='label', featuresCol='features')
lr_grid = (ParamGridBuilder() \
            .baseOn({pipeline.stages: [lr]}) \
            .addGrid(lr.regParam, [0.001, 0.01, 0.1]) \
            .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])\
            .build())
lr_cv = CrossValidator(estimator = pipeline,
                    estimatorParamMaps = lr_grid, 
                    evaluator = evaluator2, 
                    numFolds = 3,
                    parallelism = 10)
lr_cvModel = lr_cv.fit(sample)

#Extract the best configuration
best_lr = lr_cvModel.bestModel.stages[-1]._java_obj
lr_regParam = best_lr.getRegParam()
lr_elasticNetParam = best_lr.getElasticNetParam()
lr = LogisticRegression(labelCol = 'label', featuresCol = 'features',
                        regParam = lr_regParam, elasticNetParam = lr_elasticNetParam )

#create dtr with the best configuration
lr_s_time = time.time()
lr_model = lr.fit(train)
lr_e_time = time.time()
lr_predictions = lr_model.transform(test)
lr_auc = evaluator1.evaluate(lr_predictions)
lr_accuracy = evaluator2.evaluate(lr_predictions)
print("AUC of LogisticRegression = %g " % lr_auc)
print("Accuracy of LogisticRegression = %g " % lr_accuracy)
lr_imp_feat = list(lr_model.coefficients)
indice = nlargest(3, range(len(lr_imp_feat)), key=lambda i: lr_imp_feat[i])
print([feat_names[i] for i in indice])
lr_time = (lr_e_time - lr_s_time)
print("Learning time of lr = ", lr_time)
#################################################################################################
###Execution time for training
#training_time = dtc_time + dtr_time + lr_time
#print('Training time : ', training_time)
#################################################################################################END
