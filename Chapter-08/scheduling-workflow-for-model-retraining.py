# Databricks notebook source
# MAGIC %md
# MAGIC ## Training Workflow
# MAGIC
# MAGIC In this notebook, we'll create a workflow to retrain our model. Then, we'll set up this notebook to run monthly using a Databricks Job to ensure our model is always up-to-date.
# MAGIC
# MAGIC ### Load Features
# MAGIC
# MAGIC First, we'll load in our feature table which in this case is the original raw dataset.
# MAGIC
# MAGIC
# MAGIC In the case of this demonstration, these are the same records &mdash; but in real-world scenario, we'd likely have updated records appended to this table each time the model is trained.

# COMMAND ----------

# MAGIC %pip install databricks-registry-webhooks

# COMMAND ----------

database_name = "bank_churn_analysis"

#we will exclude the same columns that we did earlier while training our model using AutoML from UI.
excluded_featured_from_raw = {"RowNumber", "CustomerId", "Surname"}
target_column = "Exited"

new_data = spark.table(f"{database_name}.raw_data")
features = [c for c in new_data.columns if c not in excluded_featured_from_raw]

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Add webhook for kicking off automated testing job

# COMMAND ----------

# get token from notebook
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

#create authorization header for REST calls
headers = {"Authorization": f"Bearer {token}"}

# Next we need an enpoint at which to execute our request which we can get from the Notebook's tags collection
java_tags = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags()

# This object comes from the Java CM
tags = sc._jvm.scala.collection.JavaConversions.mapAsJavaMap(java_tags)

# extract the databricks instance (domain name) from the dictionary
instance = tags["browserHostName"]

# COMMAND ----------

model_name = "Churn Prediction Bank"

# COMMAND ----------

from databricks_registry_webhooks import RegistryWebhooksClient, JobSpec

job_spec = JobSpec(
  job_id="295266394513960",
  workspace_url="https://"+instance,
  access_token=token
)

job_webhook = RegistryWebhooksClient().create_webhook(
  model_name=model_name,
  events=["TRANSITION_REQUEST_TO_STAGING_CREATED"],
  job_spec=job_spec,
  description="Registering webhook to automate testing of a new candidate model for staging"
)

job_webhook

# COMMAND ----------

# Test the Job webhook
# RegistryWebhooksClient().test_webhook(id=job_webhook.id)

# COMMAND ----------

# MAGIC %md
# MAGIC ### AutoML Process
# MAGIC
# MAGIC Next, we'll use the AutoML API to kick off an AutoML classification experiment. This is similar to what we did with the AutoML UI, but we can use the API to automate this process.

# COMMAND ----------

import databricks.automl
model = databricks.automl.classify(
    new_data.select(features), 
    target_col=target_column,
    primary_metric="f1",
    timeout_minutes=5,
    max_trials=30,
) 

# COMMAND ----------

#information about the latest automl model training
help(model)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register the Best Model
# MAGIC
# MAGIC Once the AutoML experiment is done, we can identify the best model from the experiment and register that model to the Model Registry.

# COMMAND ----------

import mlflow
from mlflow.tracking.client import MlflowClient

client = MlflowClient()

run_id = model.best_trial.mlflow_run_id

model_uri = f"runs:/{run_id}/model"

model_details = mlflow.register_model(model_uri, model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Request model Transition to Staging
# MAGIC
# MAGIC Once the model is registered, we request that it be transitioned to the **Staging** stage for testing.
# MAGIC
# MAGIC First, we'll includ a helper function to interact with the MLflow registry API. In your production environment its always a good practice to modularize your code for maintainability.

# COMMAND ----------

# MAGIC %run ./mlflow-util

# COMMAND ----------

# MAGIC %md
# MAGIC Next, we'll set up the transition request using the `mlflow_endpoint` operation from the helpers notebook.

# COMMAND ----------

staging_request = {'name': model_name, 'version': model_details.version, 'stage': 'Staging', 'archive_existing_versions': 'false'}
mlflow_endpoint('transition-requests/create', 'POST', json.dumps(staging_request))

# COMMAND ----------

# MAGIC %md
# MAGIC And we'll add a comment to the version of the model that we just requested be moved to **Staging** to let the machine learning engineer know why we are making the request.

# COMMAND ----------

# Leave a comment for the ML engineer who will be reviewing the tests
comment = "This was the best model from the most recent AutoML run. Ready for testing"
comment_body = {'name': model_name, 'version': model_details.version, 'comment': comment}
mlflow_endpoint('comments/create', 'POST', json.dumps(comment_body))

# COMMAND ----------


