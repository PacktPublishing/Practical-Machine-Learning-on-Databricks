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
# MAGIC     - On `AWS`, select `i3.xlarge` / on `Azure`, select `Standard_DS4_V2` as __Node type__.
# MAGIC     - Click on `Create Cluster` and wait for your cluster to be provisioned.
# MAGIC
# MAGIC 2. **Attach this Notebook to Your Cluster**: 
# MAGIC     - Click on the menu labeled `Detached` at the top left of this workbook.
# MAGIC     - Select your cluster name to attach this notebook to your cluster.

# COMMAND ----------

# MAGIC %pip install assertpy

# COMMAND ----------

# the name of model in model registry you want to serve with serving endpoint.
model_name = "Churn Prediction Bank"

# serving endpoint name
model_serving_endpoint_name = "churn_prediction_api_deployment"

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC # Code Documentation for Token and Header Setup in Databricks
# MAGIC
# MAGIC This document provides an in-depth overview of the code that fetches the API token from a Databricks notebook, sets up the authorization header for REST API calls, and retrieves the Databricks instance URL.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Code Sections
# MAGIC
# MAGIC ### 1. Fetch API Token from Databricks Notebook
# MAGIC
# MAGIC #### Purpose
# MAGIC
# MAGIC - Fetches the Databricks API token from the current notebook's context.
# MAGIC
# MAGIC #### Code Explanation
# MAGIC
# MAGIC - `token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)`
# MAGIC
# MAGIC #### Libraries Used
# MAGIC
# MAGIC - `dbutils`: Databricks utility to interact with Databricks services.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 2. Create Authorization Headers
# MAGIC
# MAGIC #### Purpose
# MAGIC
# MAGIC - Sets up the headers required for authorization and content-type in REST API calls.
# MAGIC
# MAGIC #### Code Explanation
# MAGIC
# MAGIC - `headers = { "Authorization": f"Bearer {token}", "Content-Type": "application/json" }`
# MAGIC
# MAGIC #### Libraries Used
# MAGIC
# MAGIC - None.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 3. Fetch Databricks Instance URL
# MAGIC
# MAGIC #### Purpose
# MAGIC
# MAGIC - Retrieves the Databricks instance URL for further API calls.
# MAGIC
# MAGIC #### Code Explanation
# MAGIC
# MAGIC 1. `java_tags = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags()`: Fetches the notebook's tags as a Java object.
# MAGIC 2. `tags = sc._jvm.scala.collection.JavaConversions.mapAsJavaMap(java_tags)`: Converts the Java tags object to a Python dictionary.
# MAGIC 3. `instance = tags["browserHostName"]`: Extracts the Databricks instance (domain name) from the tags dictionary.
# MAGIC
# MAGIC #### Libraries Used
# MAGIC
# MAGIC - `dbutils`: Databricks utility.
# MAGIC - `sc._jvm.scala.collection.JavaConversions`: Scala library for Java to Python type conversion.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC

# COMMAND ----------

# get token from notebook
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

#create authorization header for REST calls
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
  }
 

# Next we need an enpoint at which to execute our request which we can get from the Notebook's tags collection
java_tags = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags()

# This object comes from the Java CM
tags = sc._jvm.scala.collection.JavaConversions.mapAsJavaMap(java_tags)

# extract the databricks instance (domain name) from the dictionary
instance = tags["browserHostName"]

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC # Code Documentation for `get_latest_model_version` Function
# MAGIC
# MAGIC This document offers a comprehensive overview of the `get_latest_model_version` function which retrieves the latest version number of a specified model from MLflow's model registry.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Function Overview
# MAGIC
# MAGIC ### `get_latest_model_version`
# MAGIC
# MAGIC Retrieves the latest version of a given model from the MLflow model registry.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Detailed Function Description
# MAGIC
# MAGIC ### Function: `get_latest_model_version`
# MAGIC
# MAGIC #### Purpose
# MAGIC
# MAGIC - Fetches the most recent version of a specified model from the MLflow model registry.
# MAGIC
# MAGIC #### Parameters
# MAGIC
# MAGIC - `model_name`: Name of the model for which the latest version is to be fetched.
# MAGIC
# MAGIC #### Process
# MAGIC
# MAGIC 1. **Import MlflowClient**: Imports the `MlflowClient` class from the `mlflow.tracking.client` module.
# MAGIC 2. **Initialize MLflow Client**: Instantiates an `MlflowClient` object.
# MAGIC 3. **Retrieve Latest Model Versions**: Uses the `get_latest_versions` method to fetch the latest versions of the model. Only considers versions in the "None" stage.
# MAGIC 4. **Iterate and Store Model Version**: Iterates through the returned model versions and extracts their version numbers.
# MAGIC 5. **Return Latest Version**: Returns the most recent version number of the model.
# MAGIC
# MAGIC #### Libraries Used
# MAGIC
# MAGIC - `mlflow.tracking.client`: Required for the `MlflowClient` class which is used to interact with the MLflow tracking server.
# MAGIC
# MAGIC ---
# MAGIC

# COMMAND ----------

# Import the MlflowClient class from the mlflow.tracking.client module
from mlflow.tracking.client import MlflowClient

# Define a function to get the latest version of a given model
def get_latest_model_version(model_name: str):
  # Instantiate an MlflowClient object
  client = MlflowClient()

  # Retrieve the latest versions of the specified model
  models = client.get_latest_versions(model_name)

  # Iterate through the returned models
  new_model_version = None
  for m in models:
    # Extract and store the version number of the model
    new_model_version = m.version

  # Return the latest version number
  return new_model_version

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Code Documentation for Model Endpoint Configuration
# MAGIC
# MAGIC This document provides an in-depth overview of the Python code that constructs a JSON configuration for creating or updating a model serving endpoint.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Code Sections
# MAGIC
# MAGIC ### 1. Import Required Libraries
# MAGIC
# MAGIC #### Purpose
# MAGIC
# MAGIC - Import the Python `requests` library for HTTP requests.
# MAGIC
# MAGIC #### Code Explanation
# MAGIC
# MAGIC - `import requests`
# MAGIC
# MAGIC #### Libraries Used
# MAGIC
# MAGIC - `requests`: Python library for HTTP operations.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 2. Define JSON Configuration for Model Endpoint
# MAGIC
# MAGIC #### Purpose
# MAGIC
# MAGIC - Creates a JSON object that holds the configuration for the model serving endpoint.
# MAGIC
# MAGIC #### Code Explanation
# MAGIC
# MAGIC 1. `"name": model_serving_endpoint_name`: Specifies the name of the model serving endpoint.
# MAGIC 2. `"config": {...}`: Holds the configuration details for the model serving endpoint.
# MAGIC 3. `"served_models": [...]`: A list of dictionaries, each representing a model to be served.
# MAGIC     - `"model_name": model_name`: The name of the model.
# MAGIC     - `"model_version": get_latest_model_version(model_name=model_name)`: Calls a function to get the latest version of the specified model.
# MAGIC     - `"workload_size": "Small"`: Sets the workload size to "Small".
# MAGIC     - `"scale_to_zero_enabled": True`: Enables the endpoint to scale to zero instances when not in use.
# MAGIC
# MAGIC #### Libraries Used
# MAGIC
# MAGIC - None.
# MAGIC
# MAGIC #### Dependencies
# MAGIC
# MAGIC - `model_serving_endpoint_name`: Variable holding the endpoint name.
# MAGIC - `model_name`: Variable holding the model name.
# MAGIC - `get_latest_model_version()`: Function that retrieves the latest model version.
# MAGIC
# MAGIC #### JSON Structure
# MAGIC
# MAGIC ```json
# MAGIC {
# MAGIC   "name": model_serving_endpoint_name,
# MAGIC   "config": {
# MAGIC     "served_models": [
# MAGIC       {
# MAGIC         "model_name": model_name,
# MAGIC         "model_version": get_latest_model_version(model_name=model_name),
# MAGIC         "workload_size": "Small",
# MAGIC         "scale_to_zero_enabled": True
# MAGIC       }
# MAGIC     ]
# MAGIC   }
# MAGIC }
# MAGIC ```
# MAGIC

# COMMAND ----------

import requests
 
my_json = {
  "name": model_serving_endpoint_name,
  "config": {
   "served_models": [{
     "model_name": model_name,
     "model_version": get_latest_model_version(model_name=model_name),
     "workload_size": "Small",
     "scale_to_zero_enabled": True
   }]
 }
}

# COMMAND ----------

my_json

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC # Code Documentation for Model Serving Endpoint Functions
# MAGIC
# MAGIC This document provides an overview of two Python functions—`func_create_endpoint` and `func_delete_model_serving_endpoint`—used for managing model serving endpoints.
# MAGIC
# MAGIC ## Function Overview
# MAGIC
# MAGIC ### `func_create_endpoint`
# MAGIC
# MAGIC This function either creates a new model serving endpoint or updates an existing one based on the provided parameters.
# MAGIC
# MAGIC ### `func_delete_model_serving_endpoint`
# MAGIC
# MAGIC This function deletes an existing model serving endpoint based on its name.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Detailed Function Descriptions
# MAGIC
# MAGIC ### Function: `func_create_endpoint`
# MAGIC
# MAGIC #### Purpose
# MAGIC
# MAGIC - Creates or updates the model serving endpoint.
# MAGIC
# MAGIC #### Parameters
# MAGIC
# MAGIC - `model_serving_endpoint_name`: Name of the model serving endpoint.
# MAGIC - `instance`: API instance URL.
# MAGIC - `headers`: HTTP headers for API requests.
# MAGIC - `my_json`: JSON configuration for the model serving endpoint.
# MAGIC
# MAGIC #### Process
# MAGIC
# MAGIC 1. **Define Endpoint URL**: Composes the URL where the endpoint is or will be hosted.
# MAGIC 2. **Check for Existing Endpoint**: Makes an HTTP GET request to check if the endpoint already exists.
# MAGIC 3. **Create or Update Endpoint**: 
# MAGIC    - If the endpoint does not exist, it creates a new one with the specified configuration.
# MAGIC    - If the endpoint does exist, it updates the configuration.
# MAGIC 4. **Poll for Configuration Activation**: Waits until the new configuration is active. Stops waiting after a pre-defined time (10 minutes).
# MAGIC 5. **Status Code Verification**: Checks that the API call was successful.
# MAGIC
# MAGIC #### Libraries Used
# MAGIC
# MAGIC - `requests`: For making HTTP calls.
# MAGIC - `time`: For adding sleep functionality.
# MAGIC - `json`: For JSON parsing.
# MAGIC - `assertpy`: For assertions.
# MAGIC
# MAGIC ### Function: `func_delete_model_serving_endpoint`
# MAGIC
# MAGIC #### Purpose
# MAGIC
# MAGIC - Deletes an existing model serving endpoint.
# MAGIC
# MAGIC #### Parameters
# MAGIC
# MAGIC - `model_serving_endpoint_name`: Name of the model serving endpoint.
# MAGIC - `instance`: API instance URL.
# MAGIC - `headers`: HTTP headers for API requests.
# MAGIC
# MAGIC #### Process
# MAGIC
# MAGIC 1. **Define Endpoint URL**: Composes the URL where the endpoint is hosted.
# MAGIC 2. **Delete Endpoint**: Makes an HTTP DELETE request to remove the endpoint.
# MAGIC 3. **Status Verification**: Checks if the deletion was successful and raises an exception if it fails.
# MAGIC
# MAGIC #### Libraries Used
# MAGIC
# MAGIC - `requests`: For making HTTP calls.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC This should give a detailed explanation of what each function is doing and how it accomplishes its goals.

# COMMAND ----------

import requests
import time
import json
import assertpy

def func_create_endpoint(model_serving_endpoint_name, instance, headers, my_json):
    """
    Create or update the model serving endpoint.
    """

    # Define the endpoint URL
    endpoint_url = f"https://{instance}/api/2.0/serving-endpoints"
    url = f"{endpoint_url}/{model_serving_endpoint_name}"

    # Check if the endpoint already exists
    r = requests.get(url, headers=headers)
    if "RESOURCE_DOES_NOT_EXIST" in r.text:
        print(f"Creating new endpoint: ",f"https://{instance}/serving-endpoints/{model_serving_endpoint_name}/invocations")
        re = requests.post(headers=headers, url=endpoint_url, json=my_json)
    else:
        # Extract the new model version from the JSON configuration
        new_model_version = my_json['config']['served_models'][0]['model_version']
        print(f"This endpoint existed previously! Updating it to new model version: {new_model_version}")

        # Update endpoint with new config
        url = f"{endpoint_url}/{model_serving_endpoint_name}/config"
        re = requests.put(url, headers=headers, json=my_json['config'])

        # Poll until the new configuration is active
        total_wait = 0
        while True:
            r = requests.get(url, headers=headers)
            assertpy.assert_that(r.status_code).is_equal_to(200)

            endpoint = json.loads(r.text)
            if "pending_config" in endpoint.keys():
                seconds = 10
                print("New config still pending")
                if total_wait < 600:  # 10 minutes
                    print(f"Waiting for {seconds} seconds. Total wait time: {total_wait} seconds.")
                    time.sleep(seconds)
                    total_wait += seconds
                else:
                    print(f"Stopping after {total_wait} seconds of waiting.")
                    break
            else:
                print("New config in place now!")
                break
    # Check the response code
    assertpy.assert_that(re.status_code).is_equal_to(200)

def func_delete_model_serving_endpoint(model_serving_endpoint_name, instance, headers):
    """
    Delete the model serving endpoint.
    """

    # Define the endpoint URL
    endpoint_url = f"https://{instance}/ajax-api/2.0/serving-endpoints"
    url = f"{endpoint_url}/{model_serving_endpoint_name}"

    # Delete the endpoint
    response = requests.delete(url, headers=headers)

    if response.status_code != 200:
        raise Exception(f"Request failed with status {response.status_code}, {response.text}")
    else:
        print(f"{model_serving_endpoint_name} endpoint is deleted!")


# COMMAND ----------

func_create_endpoint(model_serving_endpoint_name, instance, headers, my_json)
