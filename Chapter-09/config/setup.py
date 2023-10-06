# Databricks notebook source
# MAGIC %md
# MAGIC #### Model Drift monitoring on Databricks
# MAGIC
# MAGIC **Requirements**
# MAGIC * The following notebook was developed and tested using [DBR 13.3 LTS ML](https://docs.databricks.com/en/release-notes/runtime/13.3lts-ml.html)
# MAGIC
# MAGIC **Authors**
# MAGIC - Debu Sinha | debusinha2009@gmail.com / debu.sinha@databricks.com

# COMMAND ----------

# MAGIC %md
# MAGIC #1) Setup

# COMMAND ----------

#import mlflow if exists else install notebook scoped libraries
try:
    import mlflow
except Exception as e:
    %pip install mlflow    

# COMMAND ----------

# Get Databricks workspace username
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply("user")
print(username)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.1)  Setup Directory structure to store this demo related artifacts

# COMMAND ----------

# Set home directory for our project
project_home_dir = f"/Users/{username}/model_drift/"

#set location for temporary files created in this module
project_local_tmp_dir = f"/dbfs{project_home_dir}tmp/"

#this is where we will store raw data in csv format
raw_good_data_path= f"{project_home_dir}data/raw/good"

#this is location where data for showcasing scenario 1 for feature drift and bug in the the upstream data processing
raw_month2_bad_data_path = f"{project_home_dir}data/raw/bad"

#this is location for delta table where we will store the gold dataset
months_gold_path = f"{project_home_dir}delta/gold"

dbutils.fs.rm(project_home_dir, True)
dbutils.fs.rm(project_local_tmp_dir, True)

#reset folders for data storage
for path in [raw_good_data_path, raw_month2_bad_data_path, months_gold_path]:
    print(f"creating {path}")
    dbutils.fs.mkdirs(path)

# COMMAND ----------

# MAGIC %fs
# MAGIC ls /Users/debu.sinha@databricks.com/model_drift/data/

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.2) MLflow experiment setup

# COMMAND ----------

mlflow_experiment_name = "sales_prediction"

#this has to be an absolute path in the databricks workspace.
mlflow_experiment_path = f"/Users/{username}/{mlflow_experiment_name}"

# COMMAND ----------

import mlflow

# We need to get the exact path of experiment
experiment = mlflow.get_experiment_by_name(mlflow_experiment_path)

if experiment:
    experiment_id = experiment.experiment_id
    mlflow.delete_experiment(experiment_id)
    print(f"Experiment {mlflow_experiment_name} deleted successfully.")
    
# Create a new experiment with the specified name
experiment_id = mlflow.create_experiment(mlflow_experiment_path)
print(f"Experiment {mlflow_experiment_path} created successfully with ID {experiment_id}.")

#set the experment for this module
mlflow.set_experiment(mlflow_experiment_path)
