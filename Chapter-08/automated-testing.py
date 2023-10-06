# Databricks notebook source
# MAGIC %md 
# MAGIC Load the model name. The **`event_message`** is automatically populated by the webhook.

# COMMAND ----------


import json
 
event_message = dbutils.widgets.get("event_message")
event_message_dict = json.loads(event_message)
model_name = event_message_dict.get("model_name")

print(event_message_dict)
print(model_name)

# COMMAND ----------

# MAGIC  %md Use the model name to get the latest model version.

# COMMAND ----------

# MAGIC %run ./mlflow-util

# COMMAND ----------

from mlflow.tracking import MlflowClient
client = MlflowClient()

version = event_message_dict.get("version")
version

# COMMAND ----------

# MAGIC %md Use the model name and version to load a **`pyfunc`** model of our model in staging environment.

# COMMAND ----------

import mlflow

pyfunc_model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{version}")

# COMMAND ----------

# MAGIC %md Get the input schema of our logged model.

# COMMAND ----------

input_schema = pyfunc_model.metadata.get_input_schema().as_spark_schema()

# COMMAND ----------

# MAGIC %md Here we define our expected input schema.

# COMMAND ----------

from pyspark.sql.types import StringType, StructField, IntegerType, DoubleType, StructType

expected_input_schema = (StructType([
    StructField("CreditScore", IntegerType(), True),
    StructField("Geography", StringType(), True),
    StructField("Gender", StringType(), True),
    StructField("Age", IntegerType(), True),
    StructField("Tenure", IntegerType(), True),
    StructField("Balance", DoubleType(), True),
    StructField("NumOfProducts", IntegerType(), True),
    StructField("HasCrCard", IntegerType(), True),
    StructField("isActiveMember", IntegerType(), True),
    StructField("EstimatedSalary", DoubleType(), True)
]))

# COMMAND ----------

if expected_input_schema.fields.sort(key=lambda x: x.name) != input_schema.fields.sort(key=lambda x: x.name):
    comment = "This model failed input schema check"
    comment_body = {'name': model_name, 'version': model_details.version, 'comment': comment}
    mlflow_endpoint('comments/create', 'POST', json.dumps(comment_body))
    raise Exception("Input Schema mismatched")

# COMMAND ----------

# MAGIC %md Load the dataset and generate some predictions to ensure our model is working correctly. 

# COMMAND ----------

import pandas as pd

sample_data = spark.table("bank_churn_analysis.raw_data")
#read the raw dataset provided with the code base
df = sample_data.toPandas()

#exclude the columns that are not used for prediction
excluded_columns = {"RowNumber", "CustomerId", "Surname"}
df_input = df[[col for col in df.columns if col not in excluded_columns]]

df_input.head()

# COMMAND ----------

predictions = pyfunc_model.predict(df_input)

# COMMAND ----------

# MAGIC %md Make sure our prediction types are correct.

# COMMAND ----------

import numpy as np

if type(predictions) != np.ndarray or type(predictions[0]) != np.int32:
    comment = "This model prediction check"
    comment_body = {'name': model_name, 'version': model_details.version, 'comment': comment}
    mlflow_endpoint('comments/create', 'POST', json.dumps(comment_body))
    raise Exception("Prediction Datatype is not as expected")

# COMMAND ----------

# Leave a comment for the ML engineer who will be reviewing the tests
comment = "This model passed all the tests"
comment_body = {'name': model_name, 'version': version, 'comment': comment}
mlflow_endpoint('comments/create', 'POST', json.dumps(comment_body))
