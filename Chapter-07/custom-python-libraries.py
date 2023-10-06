# Databricks notebook source
# MAGIC %md
# MAGIC ## Author
# MAGIC
# MAGIC - **Debu Sinha**
# MAGIC
# MAGIC ## Tested Environment
# MAGIC
# MAGIC - **Databricks Runtime**: This notebook is tested on Databricks Runtime for Machine Learning 13.3 LTS or above.
# MAGIC - **Cluster Configuration**: Single node cluster with at least 32GB RAM and 4 VCPU.
# MAGIC - **Note**: The same cluster set up in Chapters 3 and 4 will be used here.
# MAGIC
# MAGIC ## Cluster Setup Instructions
# MAGIC
# MAGIC 1. **Create a Cluster**: 
# MAGIC     - Navigate to the `Compute` icon on the left sidebar and click on `Create Cluster`.
# MAGIC     - Under `Policy`, select `Unrestricted`.
# MAGIC     - Enter a name for your cluster, for example, `demo`, into the cluster name text box.
# MAGIC     - In `Cluster Mode`, select `Single Node`.
# MAGIC     - Choose `Databricks Runtime Version` 13.3 LTS (Scala 2.12, Spark 3.4.1) from the `ML` tab.
# MAGIC     - Click on `Create Cluster` and wait for your cluster to be provisioned.
# MAGIC
# MAGIC 2. **Attach this Notebook to Your Cluster**: 
# MAGIC     - Click on the menu labeled `Detached` at the top left of this workbook.
# MAGIC     - Select your cluster name to attach this notebook to your cluster.

# COMMAND ----------

# Import necessary libraries and modules
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from pandas import DataFrame
from mlflow.models import infer_signature
from mlflow.models.utils import add_libraries_to_model

# Initialize the MLflow run
with mlflow.start_run() as run:
    # Load the Iris dataset
    iris_data = load_iris()
    training_data = DataFrame(data=iris_data.data, columns=iris_data.feature_names)
    
    # Initialize and train the RandomForest Classifier
    random_forest_model = RandomForestClassifier(max_depth=7, random_state=42)
    random_forest_model.fit(training_data, iris_data.target)
    
    # Infer model signature for later use
    model_signature = infer_signature(training_data, random_forest_model.predict(training_data))
    
    # Log the trained model to MLflow
    mlflow.sklearn.log_model(random_forest_model, "iris_classifier",
                             signature=model_signature, 
                             registered_model_name="enhanced_model_with_libraries")

# Model URI for accessing the registered model
access_model_uri = "models:/enhanced_model_with_libraries/1"

# Add libraries to the original model run
add_libraries_to_model(access_model_uri)

# Example to add libraries to an existing run
# prev_run_id = "some_existing_run_id"
# add_libraries_to_model(access_model_uri, run_id=prev_run_id)


# Example to add libraries to a new run
with mlflow.start_run():
    add_libraries_to_model(access_model_uri)

# Example to add libraries and register under a new model name
with mlflow.start_run():
    add_libraries_to_model(access_model_uri, registered_model_name="new_enhanced_model")
