# Databricks notebook source
# MAGIC %md # MLflow introduction.
# MAGIC
# MAGIC This tutorial covers an example of how to use the integrated MLflow tracking capabilities to track your model training with the integrated feature store.
# MAGIC   - Import data that was previously registered in the feature store table.
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
# MAGIC   - Click the `create cluster` button and wait for your cluster to be provisioned
# MAGIC 3. **Attach** this notebook to your cluster by...   
# MAGIC   - Click on your cluster name in menu `Detached` at the top left of this workbook to attach it to this workbook 

# COMMAND ----------

#install latest version of sklearn
%pip install -U scikit-learn

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Step 1) Importing the desired libraries and defining few constants and creating training set from the registered feature table.

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

#Name of the model
MODEL_NAME = "random_forest_classifier_featurestore"
#This is the name for the entry in model registry
MODEL_REGISTRY_NAME = "Bank_Customer_Churn"
#The email you use to authenticate in the Databricks workspace
USER_EMAIL = "debu.sinha@databricks.com"
#Location where the MLflow experiement will be listed in user workspace
EXPERIMENT_NAME = f"/Users/{USER_EMAIL}/Bank_Customer_Churn_Analysis"
# we have all the features backed into a Delta table so we will read directly
FEATURE_TABLE = "bank_churn_analysis.bank_customer_features"


# COMMAND ----------


# this code is just for demonstration and you can utilize this as starting point and build more errorhandling around it.
class Feature_Lookup_Input_Tuple(typing.NamedTuple):
  fature_table_name: str
  feature_list: typing.Union[typing.List[str], None] 
  lookup_key: typing.List[str]

# this code is going to generate feature look up based on on the list of feature mappings provided.
def generate_feature_lookup(feature_mapping: typing.List[Feature_Lookup_Input_Tuple]) -> typing.List[FeatureLookup]:  
  lookups = []
  for fature_table_name, feature_list, lookup_key in feature_mapping:
    lookups.append(
          FeatureLookup(
          table_name = fature_table_name,
          feature_names = feature_list,
          lookup_key = lookup_key 
      )
    )
  return lookups


# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2) Build a simplistic model that uses the feature store table as its source for training and validation.

# COMMAND ----------

#initialize the feature store client
fs = FeatureStoreClient()
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run():  
  TEST_SIZE = 0.20
  
  #define the list of features we want to get from feature table
  #If we have to combine data from multiple feature tables then we can provide multiple mappings for feature tables 
  features = [Feature_Lookup_Input_Tuple(FEATURE_TABLE,["CreditScore" , "Age", "Tenure",\
              "Balance", "NumOfProducts", "HasCrCard",\
              "IsActiveMember", "EstimatedSalary", "Geography_Germany",\
              "Geography_Spain", "Gender_Male"], ["CustomerId"] )]

  lookups = generate_feature_lookup(features)
  
  #Now we will simulate receiving only ID's of customers and the label as input at the  time of inference
  training_df = spark.table(FEATURE_TABLE).select("CustomerId", "Exited")
  
  #Using the training set we will combine the training dataframe with the features stored in the feature tables.
  training_data = fs.create_training_set(
    df=training_df,
    feature_lookups=lookups,
    label="Exited",
    exclude_columns=['CustomerId']
  )
  
  #convert the dataset to pandas so that we can fit sklearn RandomForestClassifier on it
  train_df = training_data.load_df().toPandas()
  
  #The train_df represents the input dataframe that has all the feature columns along with the new raw input in the form of training_df.
  X = train_df.drop(['Exited'], axis=1)
  y = train_df['Exited']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=54, stratify=y)
  
  #here we will are not doing any hyperparameter tuning however, in future we will see how to perform hyperparameter tuning in scalable manner on Databricks.
  model = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)
  signature = mlflow.models.signature.infer_signature(X_train, model.predict(X_train))
  
  predictions = model.predict(X_test)
  fpr, tpr, _ = metrics.roc_curve(y_test, predictions, pos_label=1)
  auc = metrics.auc(fpr, tpr)
  accuracy = metrics.accuracy_score(y_test, predictions)
 
  #get the calculated feature importances.
  importances = dict(zip(model.feature_names_in_, model.feature_importances_))  
  #log artifact
  mlflow.log_dict(importances, "feature_importances.json")
  #log metrics
  mlflow.log_metric("auc", auc)
  mlflow.log_metric("accuracy", accuracy)
  #log parameters
  mlflow.log_param("split_size", TEST_SIZE)
  mlflow.log_params(model.get_params())
  #set tag
  mlflow.set_tag(MODEL_NAME, "mlflow and feature store demo")
  #log the model itself in mlflow tracking server
  mlflow.sklearn.log_model(model, MODEL_NAME, signature=signature, input_example=X_train.iloc[:4, :])

  # finally to make the feature store track what features are being used by our model we call log_model with the feature store client
  fs.log_model(
    model,
    MODEL_NAME,
    flavor=mlflow.sklearn,
    training_set=training_data,
    registered_model_name=MODEL_REGISTRY_NAME
  )
  
  

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3) Now that we have the model logged to the MLflow tracking server, we can get the latest version from the experiment and use it.

# COMMAND ----------

from mlflow.tracking import MlflowClient
#initialize the mlflow client
client = MlflowClient()
#get the experiment id 
experiment_id = mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id
#get the latest run id which will allow us to directly access the metrics, and attributes and all th einfo
run_id = mlflow.search_runs(experiment_id, order_by=["start_time DESC"]).head(1)["run_id"].values[0]

# COMMAND ----------

# MAGIC %md
# MAGIC - With the feature store registration associated with the MLflow model, we don't have to specify any data loading and processing to happen other than a point to the raw data that features will be calculated from. 
# MAGIC - We can do batch predictions simply by accessing the feature store instance, providing the run_id and the model's name (MODEL_NAME below) with the raw data specified as the second argument. 
# MAGIC - If we want to provide new values for certain feature that is already part of the feature table, just include it in the new dataframe that we want to perform the prediction on.

# COMMAND ----------

#at the time of infernce you can provide just the CustomerId. This is the key that will perform all the lookup for the features automatically.
predictions = fs.score_batch(f"runs:/{run_id}/{MODEL_NAME}", spark.table(FEATURE_TABLE).select("CustomerId"))

# COMMAND ----------

display(predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Cleanup

# COMMAND ----------

#Uncomment to lines below and execute for cleaning up.
'''
from mlflow.tracking import MlflowClient

#get all the information about the current experiment
experiment_id = mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

#list all the runs that are part of this experiment and delete them
runs = mlflow.list_run_infos(experiment_id=experiment_id)
for run in runs:
  mlflow.delete_run(run_id = run.run_id)

#finally delete the experiment  
mlflow.delete_experiment(experiment_id=experiment_id)  

client = MlflowClient()
#delete the model registered in the registry to clear the linkage in thefeature store
client.delete_registered_model(name=MODEL_REGISTRY_NAME)
'''
