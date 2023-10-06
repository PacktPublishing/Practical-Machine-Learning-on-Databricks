# Databricks notebook source
# MAGIC %md # MLflow introduction.
# MAGIC
# MAGIC This tutorial covers an example of how to use the integrated MLflow tracking capabilities to track your model training with the integrated feature store.
# MAGIC   - Import data from the Delta table that contains feature engineered datasets.
# MAGIC   - Create a baseline model for churn prediction and store it in the integrated MLflow tracking server. 

# COMMAND ----------

# MAGIC %md
# MAGIC ###0. SETUP -- Databricks Spark cluster:  
# MAGIC
# MAGIC 1. **Create** a cluster by...  
# MAGIC   - Click the `Compute` icon on the left sidebar and then `Create Cluster.` 
# MAGIC   - In `Policy` select `Unrestricted`.
# MAGIC   - Enter any text, i.e `demo` into the cluster name text box.
# MAGIC   - Select `Single Node` in the cluster mode.
# MAGIC   - Select the `Databricks runtime version` value `13.3 LTS (Scala 2.12, Spark 3.4.1)` from the `ML` tab.
# MAGIC   - On `AWS`, select `i3.xlarge` / on `Azure`, select `Standard_DS4_V2` as __Node type__.
# MAGIC   - Click the `create cluster` button and wait for your cluster to be provisioned
# MAGIC 3. **Attach** this notebook to your cluster by...   
# MAGIC   - Click on your cluster name in menu `Detached` at the top left of this workbook to attach it to this workbook 

# COMMAND ----------

#install latest version of sklearn
%pip install -U scikit-learn

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Step 1) Importing the desired libraries and defining few constants.
# MAGIC
# MAGIC - Note:<br>
# MAGIC   - In this example the feature table is the same as we created in Chapter 3, however we will not use the featurestore API to access the data in the feature table.<br>
# MAGIC   - As explained in chapter 3, all the offline feature tables are backed as Delta tables and are searchable through the integrated Hive metastore in Databricks. This allows us to read these tables like a regular external or managed table.

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient
from databricks.feature_store import FeatureLookup
import typing

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mlflow
import pandas as pd

# COMMAND ----------

#Name of experiment where we will track all the different model training runs.
EXPERIMENT_NAME = "Bank_Customer_Churn_Analysis"
#Name of the model
MODEL_NAME = "random_forest_classifier"
#This is the name for the entry in model registry
MODEL_REGISTRY_NAME = "Bank_Customer_Churn"
#The email you use to authenticate in the Databricks workspace
USER_EMAIL = "debu.sinha@databricks.com"
#Location where the MLflow experiement will be listed in user workspace
EXPERIMENT_NAME = f"/Users/{USER_EMAIL}/{EXPERIMENT_NAME}"
# we have all the features backed into a Delta table so we will read directly
FEATURE_TABLE = "bank_churn_analysis.bank_customer_features"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2) Build a simplistic model that uses the feature store table as its source for training and validation.

# COMMAND ----------

# set experiment name
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run():  
  TEST_SIZE = 0.20
  
  # Now we will read the data directly from the feature table
  training_df = spark.table(FEATURE_TABLE)
  
  # convert the dataset to pandas so that we can fit sklearn RandomForestClassifier on it
  train_df = training_df.toPandas()
  
  # The train_df represents the input dataframe that has all the feature columns along with the new raw input in the form of training_df.
  X = train_df.drop(['Exited'], axis=1)
  y = train_df['Exited']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=54, stratify=y)
  
  # here we will are not doing any hyperparameter tuning however, in future we will see how to perform hyperparameter tuning in scalable manner on Databricks.
  model = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)
  signature = mlflow.models.signature.infer_signature(X_train, model.predict(X_train))
  
  predictions = model.predict(X_test)
  fpr, tpr, _ = metrics.roc_curve(y_test, predictions, pos_label=1)
  auc = metrics.auc(fpr, tpr)
  accuracy = metrics.accuracy_score(y_test, predictions)
 
  # get the calculated feature importances.
  importances = dict(zip(model.feature_names_in_, model.feature_importances_))  
  # log artifact
  mlflow.log_dict(importances, "feature_importances.json")
  # log metrics
  mlflow.log_metric("auc", auc)
  mlflow.log_metric("accuracy", accuracy)
  # log parameters
  mlflow.log_param("split_size", TEST_SIZE)
  mlflow.log_params(model.get_params())
  # set tag
  mlflow.set_tag(MODEL_NAME, "mlflow demo")
  # log the model itself in mlflow tracking server
  mlflow.sklearn.log_model(model, MODEL_NAME, signature=signature, input_example=X_train.iloc[:4, :])

# COMMAND ----------

from mlflow.tracking import MlflowClient
#initialize the mlflow client
client = MlflowClient()
#get the experiment id 
experiment_id = mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id
#get the latest run id which will allow us to directly access the metrics, and attributes and all th einfo
run_id = mlflow.search_runs(experiment_id, order_by=["start_time DESC"]).head(1)["run_id"].values[0]
#now we will register the latest model into the model registry
new_model_version = mlflow.register_model(f"runs:/{run_id}/{MODEL_NAME}", MODEL_REGISTRY_NAME)
