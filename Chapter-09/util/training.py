# Databricks notebook source
from delta.tables import DeltaTable
import tempfile
import os
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyspark
import pyspark.sql.functions as F

import math
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_selector as selector
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec, DataType

# COMMAND ----------

#mlflow util functions to manage models
def transition_model(model_version, stage):
    """
    Transition a model to a specified stage in MLflow Model Registry using the associated 
    mlflow.entities.model_registry.ModelVersion object.

    Args:
        model_version: mlflow.entities.model_registry.ModelVersion. ModelVersion object to transition
        stage: (str) New desired stage for this model version. One of "Staging", "Production", "Archived" or "None"

    Returns:
        A single mlflow.entities.model_registry.ModelVersion object
    """
    client = MlflowClient()

    # Check if the stage is valid
    if stage not in ["Staging", "Production", "Archived", "None"]:
        raise ValueError(f"Invalid stage: {stage}")

    # Transition the model version
    model_version = client.transition_model_version_stage(
        name=model_version.name,
        version=model_version.version,
        stage=stage,
        archive_existing_versions=True,
    )

    return model_version


def fetch_model_version(registry_model_name, stage="Staging"):
    """
    For a given registered model, return the MLflow ModelVersion object
    This contains all metadata needed, such as params logged etc

    Args:
        registry_model_name: (str) Name of MLflow Registry Model
        stage: (str) Stage for this model. One of "Staging" or "Production"

    Returns:
        mlflow.entities.model_registry.ModelVersion
    """
    client = MlflowClient()
    filter_string = f'name="{registry_model_name}"'
    registered_model = client.search_registered_models(filter_string=filter_string)[0]

    # Check if the stage is valid
    if stage not in ["Staging", "Production"]:
        raise ValueError(f"Invalid stage: {stage}")

    # Get the latest model version in the desired stage
    model_version = next(
        (model_version for model_version in registered_model.latest_versions if model_version.current_stage == stage),
        None
    )

    return model_version


def get_run_from_registered_model(registry_model_name, stage="Staging"):
    """
    Get Mlflow run object from registered model

    Args:
        registry_model_name: (str) Name of MLflow Registry Model
        stage: (str) Stage for this model. One of "Staging" or "Production"

    Returns:
        mlflow.entities.run.Run
    """
    client = MlflowClient()
    filter_string = f'name="{registry_model_name}"'
    registered_model = client.search_registered_models(filter_string=filter_string)[0]

    # Check if the stage is valid
    if stage not in ["Staging", "Production"]:
        raise ValueError(f"Invalid stage: {stage}")

    # Get the latest model version in the desired stage
    model_version = next(
        (model_version for model_version in registered_model.latest_versions if model_version.current_stage == stage),
        None
    )

    if model_version is None:
        raise ValueError(f"No model version found in stage {stage} for model {registry_model_name}")

    run_id = model_version.run_id
    run = mlflow.get_run(run_id)

    return run


def cleanup_registered_model(registry_model_name: str) -> None:
    """
    Deletes a registered model in MLflow model registry.

    To delete a model in the model registry, all model versions must first be archived.
    This function first archives all versions of a model in the registry, and then deletes the model.

    Args:
        registry_model_name: The name of the model in the MLflow model registry.
    """
    client = MlflowClient()

    filter_string = f'name="{registry_model_name}"'

    model_versions = client.search_model_versions(filter_string=filter_string)

    if len(model_versions) == 0:
        logging.info("No registered models to delete")
        return

    logging.info(f"Deleting following registered model: {registry_model_name}")

    # Move any versions of the model to Archived
    for model_version in model_versions:
        try:
            if model_version.current_stage!='Archived':
                client.transition_model_version_stage(
                    name=model_version.name,
                    version=model_version.version,
                    stage="Archived",
                )
        except Exception as e:
            logging.exception(f"Error archiving version {model_version.version} of model {registry_model_name}")
            raise

    try:
        client.delete_registered_model(registry_model_name)
    except RestException as e:
        logging.exception(f"Error deleting registered model {registry_model_name}")
        raise



# COMMAND ----------

#delete any registered model from registry
cleanup_registered_model(mlflow_experiment_name)

# COMMAND ----------


#delta table utility functions
def get_delta_version(delta_path: str) -> int:
    """
    Gets the latest version of a Delta table given the path to the table.

    Args:
        delta_path: The path to the Delta table

    Returns:
        The version of the Delta table.
    """
    try:
        delta_table = DeltaTable.forPath(spark, delta_path)
        delta_history= delta_table.history()

         # Retrieve the lastest Delta version - this is the version loaded when reading from delta_path
        delta_version = delta_history.first()["version"]
  
        return delta_version

    except AnalysisException as e:
        raise ValueError(f"Error getting Delta table version: {e}")

def load_delta_table_from_run(run: mlflow.entities.run.Run) -> pyspark.sql.DataFrame:
    """
    Given an MLflow run, load the Delta table which was used for that run,
    using the path and version tracked at tracking time.

    Note that by default Delta tables only retain a commit history for 30 days, meaning
    that previous versions older than 30 days will be deleted by default. This property can
    be updated using the Delta table property `delta.logRetentionDuration`.

    For more information, see https://docs.databricks.com/delta/delta-batch.html#data-retention

    Args:
        run: The MLflow run object.

    Returns:
        The Spark DataFrame for the Delta table used in the run.
    """
    delta_path = run.data.params.get("delta_path")
    delta_version = run.data.params.get("delta_version")
    if not delta_path or not delta_version:
        raise ValueError("Error: missing delta_path or delta_version parameters.")
    print(f"Loading Delta table from path: {delta_path}; version: {delta_version}")
    try:
        df = spark.read.format("delta").option("versionAsOf", delta_version).load(delta_path)
        return df
    except Exception as e:
        print(f"Error: could not load Delta table. {str(e)}")
        raise

# COMMAND ----------

def calculate_summary_stats(pdf: pd.DataFrame) -> pd.DataFrame:
  """
  Create a pandas DataFrame of summary statistics for a provided pandas DataFrame.
  Involved calling .describe on pandas DataFrame provided and additionally add
  median values and a count of null values for each column.
  
  :param pdf: pandas DataFrame
  :return: pandas DataFrame of sumary statistics for each column
  """
  stats_pdf = pdf.describe(include="all")

  # Add median values row
  median_vals = pdf.median()
  stats_pdf.loc["median"] = median_vals

  # Add null values row
  null_count = pdf.isna().sum()
  stats_pdf.loc["null_count"] = null_count

  return stats_pdf


def log_summary_stats_pdf_as_csv(pdf: pd.DataFrame) -> None:
    """
    Log summary statistics pandas DataFrame as a csv file to MLflow as an artifact.

    Args:
        pdf: A pandas DataFrame containing summary statistics.
    """
    with tempfile.NamedTemporaryFile(prefix="summary_stats", suffix=".csv", delete=False) as temp:
        pdf.to_csv(temp.name, index=True)
        artifact_name = "summary_stats.csv"
        shutil.move(temp.name, artifact_name)
        mlflow.log_artifact(artifact_name, artifact_path="summary_stats")
        os.remove(artifact_name)


def load_summary_stats_pdf_from_run(run: mlflow.entities.run.Run, local_tmp_dir: str) -> pd.DataFrame:
    """
    Given an MLflow run, download the summary stats csv artifact to a local_tmp_dir and load the
    csv into a pandas DataFrame.

    Args:
        run: The MLflow run object.
        local_tmp_dir: (str) path to a local filesystem tmp directory

    Returns:
        A pandas DataFrame containing statistics computed during training.
    """

    # Use mlflow to download the csv file logged in the artifacts of a run to a local tmp path
    Path(local_tmp_dir).mkdir(parents=True, exist_ok=True)
    local_path=mlflow.artifacts.download_artifacts(run_id=run.info.run_id, artifact_path="summary_stats", dst_path=local_tmp_dir)
    print(f"Summary stats artifact downloaded in: {local_path}")

    # Load the csv into a pandas DataFrame
    summary_stats_path = os.path.join(local_path, os.listdir(local_path)[0])
    try:
        summary_stats_pdf = pd.read_csv(summary_stats_path, index_col="Unnamed: 0")
    except Exception as e:
        raise ValueError(f"Failed to load summary stats csv from path {summary_stats_path}: {e}")

    return summary_stats_pdf        

# COMMAND ----------

def create_sklearn_rf_pipeline(model_params, seed=42):
    """
    Create the sklearn pipeline required for the RandomForestRegressor.
    We compose two components of the pipeline separately - one for numeric cols, one for categorical cols
    These are then combined with the final RandomForestRegressor stage, which uses the model_params dict
    provided via the args. The unfitted pipeline is returned.

    For a robust pipeline in practice, one should also have a pipeline stage to add indicator columns for those features
    which have been imputed. This can be useful to encode information about those instances which have been imputed with
    a given value. We refrain from doing so here to simplify the pipeline, and focus on the overall workflow.

    Args:
        model_params: (dict) Dictionary of model parameters to pass into sklearn RandomForestRegressor
        seed : (int) Random seed to set via random_state arg in RandomForestRegressor

    Returns:
        sklearn pipeline
    """
    # Create pipeline component for numeric Features
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy='median'))])

    # Create pipeline component for categorical Features
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))])

    # Combine numeric and categorical components into one preprocessor pipeline
    # Use ColumnTransformer to apply the different preprocessing pipelines to different subsets of features
    # Use selector (make_column_selector) to select which subset of features to apply pipeline to
    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_transformer, selector(dtype_exclude="category")),
        ("categorical", categorical_transformer, selector(dtype_include="category"))
    ])

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("rf", RandomForestRegressor(random_state=seed, **model_params))
    ])

    return pipeline

def train_sklearn_rf_model(run_name, delta_path, model_params, misc_params, seed=42):
  """
  Function to trigger training and evaluation of an sklearn RandomForestRegressor model.

  Parameters, metrics, and artifacts are logged to MLflow during this process.

  Returns the MLflow run object.

  Args:
    run_name: (str) Name to give to MLflow run.
    delta_path: (str) Path to Delta table to use as input data.
    model_params: (dict) Dictionary of model parameters to pass into sklearn RandomForestRegressor.
    misc_params: (dict) Dictionary of parameters to use.
    seed: (int) Random seed.

  Returns:
    mlflow.entities.run.Run
  """

  #end any active run 
  mlflow.end_run()
  
  # Enable MLflow autologging.
  mlflow.autolog(log_input_examples=True, silent=True)

  # Load Delta table from `delta_path`.
  df = spark.read.format("delta").load(delta_path)

  # Log `delta_path` and version.
  mlflow.log_param("delta_path", delta_path)
  delta_version = get_delta_version(delta_path)
  mlflow.log_param("delta_version", delta_version)

  # Track misc parameters used in pipeline creation (preprocessing) as JSON artifact.
  mlflow.log_dict(misc_params, "preprocessing_params.json")

  # Convert Spark DataFrame to pandas, as we will be training an sklearn model.
  pdf = df.toPandas()

  # Convert all categorical columns to category dtype.
  for c in misc_params["cat_cols"]:
    pdf[c] = pdf[c].astype("category")

  #keek only the required columns
  cols_to_keep = np.concatenate(([misc_params['target_col']], misc_params['cat_cols'], misc_params['num_cols']), axis=None)
  pdf = pdf[cols_to_keep]


  # Create summary statistics pandas DataFrame and log as a CSV to MLflow.
  summary_stats_pdf = calculate_summary_stats(pdf[cols_to_keep])
  log_summary_stats_pdf_as_csv(summary_stats_pdf)

  # Track number of total instances and "month".
  num_instances = pdf.shape[0]
  mlflow.log_param("num_instances", num_instances)  # Log number of instances.
  mlflow.log_param("month", misc_params["month"])   # Log month number.

  # Split data.
  X = pdf.drop([misc_params["target_col"]], axis=1)
  y = pdf[misc_params["target_col"]]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

  # Track train/test data info as parameters.
  num_training = X_train.shape[0]
  mlflow.log_param("num_training_instances", num_training)
  num_test = X_test.shape[0]
  mlflow.log_param("num_test_instances", num_test)

  # Fit sklearn pipeline with RandomForestRegressor model.
  rf_pipeline = create_sklearn_rf_pipeline(model_params)
  rf_pipeline.fit(X_train, y_train)

  # Make predictions on the test data
  y_pred = rf_pipeline.predict(X_test)
  
  # Calculate evaluation metrics on the test data
  mae = mean_absolute_error(y_test, y_pred)
  mse = mean_squared_error(y_test, y_pred)
  r2 = r2_score(y_test, y_pred)
  rmse = math.sqrt(mse)

  # Specify data schema which the model will use as its ModelSignature.

  input_schema = Schema([
        ColSpec(name="Weather_Condition", type=DataType.string),
        ColSpec(name="Promotion_Type", type=DataType.string),
        ColSpec(name="Device_Type", type=DataType.string),
        ColSpec(name="Temperature", type=DataType.float),
        ColSpec(name="Website_Traffic", type=DataType.integer)
   ])

  output_schema = Schema([ColSpec("integer")])
  signature = ModelSignature(input_schema, output_schema)
  mlflow.sklearn.log_model(rf_pipeline, "model", signature=signature)

  return mlflow.active_run()



