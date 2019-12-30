# -*- coding: utf-8 -*-
'''
COM6012 Scalable Machine Learning
Assignment 02
180228768
Question 2. Senior Data Analyst at ​Intelligent Insurances Co.​ ​[10 marks]
'''
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.ml import Transformer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, GeneralizedLinearRegression
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import isnull, when, count
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline, Transformer
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.ml.feature import OneHotEncoder, StandardScaler, VectorAssembler, StringIndexer, Bucketizer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, Binarizer

from pyspark.ml.regression import LinearRegression, GeneralizedLinearRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.param import Params

from pyspark.sql.functions import isnull, when, count, col
from pprint import pprint
import time
import findspark
findspark.init()
##################################################################################################
#HPC - 16CORES
spark = SparkSession.builder\
        .config("spark.locl.dir", "/fastdata/acp18hp")\
        .appName("AS02")\
        .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")
##################################################################################################
# HPC - read data
data = spark.read.option("inferschema",False).option("header", "true").option("nullValue", "?").csv("/fastdata/acp18hp/train_set.csv").repartition(10)
data = data.withColumnRenamed("Claim_Amount", "label")
data.cache()
##################################################################################################
### Preprocessing data
# drop some columns with many rows of na and a few cols
rows_cnt = data.count()
cols_na = list(data.select([count(when(isnull(c), c)).alias(c) for c in data.columns]).collect()[0])
cols_drop = [data.columns[i] for i,cnt in enumerate(cols_na) if cnt>(0.3*rows_cnt)]
columns_drop = list(set(cols_drop + ["Row_ID","Household_ID","Vehicle","Blind_Make","Blind_Model","Blind_Submodel"]))
data = data.drop(*columns_drop)
##################################################################################################
##### AMEND - THINK MORE ! TO CHANGE ANOTHER ONE NOT JUST ZERO,0
# impute some rows with na into 0
cols_na = list(data.select([count(when(isnull(c), c)).alias(c) for c in data.columns]).collect()[0])
cols_impu = [data.columns[i] for i,cnt in enumerate(cols_na) if cnt>0]
dic_cols_impu = dict.fromkeys(cols_impu, 0)
data = data.na.fill(dic_cols_impu)
##################################################################################################
# Oversampling rows with non-zero label, and Undersampling rows with zero label
def check_unbal(data, oversampling = False):
    zero = data[data.label == 0.0]
    non_zero = data[data.label != 0.0]
    pro_zero = zero.count() / data.count()
    print('before')
    print('Prob. of zero : ', pro_zero*100)
    print('Prob. of non-zero : ', (1 - pro_zero)*100)
    if(oversampling):
        # Oversampling of non-zero
        for i in range(50):
            data = data.unionAll(non_zero.sample(True, 0.99))
    zero = data[data.label == 0.0]
    pro_zero = zero.count() / data.count()
    print('after')
    print('Prob. of zero : ', pro_zero*100)
    print('Prob. of non-zero : ', (1 - pro_zero)*100)
    return data
##################################################################################################
# categorical data
cats = ["Calendar_Year","Model_Year","Cat1","Cat2","Cat3","Cat4","Cat5","Cat6","Cat7","Cat8","Cat9","Cat10","Cat11","Cat12","OrdCat","NVCat"]
cats = list(set(cats) - set(columns_drop))
# continuous data - String(but, real number) should be "Doubletype"
cont = ["Var1","Var2","Var3","Var4","Var5","Var6","Var7","Var8", "NVVar1","NVVar2","NVVar3","NVVar4"]
cont = list(set(cont) - set(columns_drop))
##################################################################################################
### Data preprocessing for feature vector
indexers = [StringIndexer(inputCol = c, outputCol = c+'_1') for c in cats]
encoder = [OneHotEncoder(inputCol = c+'_1', outputCol = c+'_2') for c in cats]
cats = [i+"_2" for i in cats]

class ParserDouble(Transformer):  
    def __init__(self, columns=[None]):
        self.columns = columns
    def _transform(self, df):
        for c in self.columns:
            df = df.withColumn(c, df[c].cast(DoubleType()))
        self = df
        return self
doubler = ParserDouble(data.columns)
assembler = VectorAssembler(inputCols = cats+cont, outputCol = 'features')
pipeline = Pipeline(stages = indexers + encoder + [doubler, assembler])

model = pipeline.fit(data)
data = model.transform(data).select('features', 'label')
##################################################################################################
evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="label", metricName="r2")
train, test = data.randomSplit([0.8, 0.2], seed=12345)
train = check_unbal(train, True)
##################################################################################################
# Q2 - a. Linear Regression

lr = LinearRegression(labelCol='label', featuresCol='features', predictionCol='prediction')
lr_s_time = time.time()
lr_model = lr.fit(train)
lr_e_time = time.time()
lr_predictions = lr_model.transform(test)
r2 = evaluator.evaluate(lr_predictions)
print("r2 of lr = %g " % r2)
lr_time = lr_e_time - lr_s_time
print("Learning time of lr = ", lr_time)
##################################################################################################
# Q2 - b. Gamma Regression
# data of non-zero
data_nz = data[data.label != 0.0]
train, test = data_nz.randomSplit([0.8, 0.2], seed=12345)
glr = GeneralizedLinearRegression(labelCol='label', featuresCol='features', 
                                                          predictionCol='prediction', family="gamma")
glr_s_time= time.time()
glr_model = glr.fit(train)
glr_e_time = time.time()
glr_predictions = glr_model.transform(test)
r2 = evaluator.evaluate(glr_predictions)
print("r2 of glr = %g " % r2)
glr_time = glr_e_time - glr_s_time
print("Learning time of glr = ", glr_time)
##################################################################################################