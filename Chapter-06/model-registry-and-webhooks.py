# Databricks notebook source
# MAGIC
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
# MAGIC
# MAGIC ## MLflow Model Registry API
# MAGIC
# MAGIC This section demonstrates how to register a model in the registry and request its transition to the staging environment.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Retrieving the Most Recently Updated Experiment from the MLflow Server
# MAGIC
# MAGIC In this code snippet, several key tasks are carried out:
# MAGIC
# MAGIC 1. **Initialize MLflow Client**: 
# MAGIC    - The MLflow tracking client is initialized to interact with the MLflow server.
# MAGIC   
# MAGIC 2. **Fetch Available Experiments**: 
# MAGIC    - A list of all available experiments is fetched using the `search_experiments()` method of the client.
# MAGIC   
# MAGIC 3. **Sort Experiments by Last Update Time**: 
# MAGIC    - The fetched experiments are sorted based on their last update time in descending order, ensuring that the most recently modified experiment comes first.
# MAGIC
# MAGIC 4. **Retrieve Latest Experiment**: 
# MAGIC    - The most recently updated experiment is then extracted from the sorted list and stored in the `latest_experiment` variable.
# MAGIC
# MAGIC 5. **Display Experiment Name**: 
# MAGIC    - The name of the most recently updated experiment is printed out for confirmation.
# MAGIC
# MAGIC > **Note**: If you are specifically interested in the experiment related to AutoML for base model creation, make sure that the `latest_experiment` actually corresponds to that particular experiment.
# MAGIC

# COMMAND ----------

import mlflow

# Initialize the MLflow client
client = mlflow.tracking.MlflowClient()

# Fetch all available experiments
experiments = client.search_experiments()

# Sort the experiments by their last update time in descending order
sorted_experiments = sorted(experiments, key=lambda x: x.last_update_time, reverse=True)

# Retrieve the most recently updated experiment
latest_experiment = sorted_experiments[0]

# Output the name of the latest experiment
print(f"The most recently updated experiment is named '{latest_experiment.name}'.")

# Note: If you're specifically looking for the experiment related to AutoML for base model creation,
# ensure that 'latest_experiment' corresponds to that experiment.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Identifying the Best Model Run ID from a Specific Experiment in MLflow
# MAGIC
# MAGIC In this code snippet, the objective is multi-fold:
# MAGIC
# MAGIC 1. **Fetch Current User's Username**: 
# MAGIC    - Utilizes Databricks utilities to programmatically fetch the username. This could be useful for traceability or logging purposes.
# MAGIC
# MAGIC 2. **Set Experiment and Model Names**: 
# MAGIC    - Retrieves the name of the most recently updated experiment, assumed to have been set in earlier steps.
# MAGIC    - Defines a specific name for the model in the registry, which in this case is "Churn Prediction Bank".
# MAGIC
# MAGIC 3. **Fetch and Sort Experiment Runs**: 
# MAGIC    - Retrieves the details of the experiment using its name.
# MAGIC    - Searches for all runs within the experiment and sorts them based on the F1 score on the validation set, in descending order.
# MAGIC
# MAGIC 4. **Identify the Best Model Run ID**: 
# MAGIC    - The run ID corresponding to the highest validation F1 score is then stored in the `best_run_id` variable.
# MAGIC
# MAGIC > **Note**: The `best_run_id` variable now holds the run ID of the model that performed best in the specified experiment, according to the F1 score on the validation set.
# MAGIC
# MAGIC

# COMMAND ----------

# Initialize the Databricks utilities to programmatically fetch the username
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()

# Retrieve the name of the latest experiment; assumed to have been set in earlier steps
experiment_name = latest_experiment.name

# Define the model name for the registry, specific to our use-case of Churn Prediction for a Bank
registry_model_name = "Churn Prediction Bank"

# Fetch the experiment details using its name
experiment_details = client.get_experiment_by_name(experiment_name)

# Search for runs within the experiment and sort them by validation F1 score in descending order
sorted_runs = mlflow.search_runs(experiment_details.experiment_id).sort_values("metrics.val_f1_score", ascending=False)

# Get the run ID of the best model based on the highest validation F1 score
best_run_id = sorted_runs.loc[0, "run_id"]

best_run_id
# Note: The variable `best_run_id` now contains the run ID of the best model in the specified experiment

# COMMAND ----------

# MAGIC %md
# MAGIC ### Registering the Best Model in MLflow's Model Registry
# MAGIC
# MAGIC The aim of this code block is to register the best-performing model (based on the highest validation F1 score) in MLflow's model registry. Here's how it does it:
# MAGIC
# MAGIC 1. **Initialize Model URI**: 
# MAGIC    - Constructs the model URI using the `best_run_id` obtained from previous steps. The URI will uniquely identify the model's location.
# MAGIC
# MAGIC 2. **Attempt Model Registration**: 
# MAGIC    - Tries to register the model under the name specified by `registry_model_name`.
# MAGIC   
# MAGIC 3. **Success and Failure Scenarios**: 
# MAGIC    - Prints a success message along with the model URI if the model registration is successful.
# MAGIC    - Captures and prints an error message if it fails to register the model.
# MAGIC
# MAGIC > **Note**: The `model_details` variable will be populated with details about the registered model if the registration is successful. These details include the model name, version, and other metadata.
# MAGIC

# COMMAND ----------

# Initialize the model's URI using the best run ID obtained from previous steps
model_uri = f"runs:/{best_run_id}/model"

# Register the model in MLflow's model registry under the specified name
try:
    model_details = mlflow.register_model(model_uri=model_uri, name=registry_model_name)
    print(f"Successfully registered model '{registry_model_name}' with URI '{model_uri}'.")
except mlflow.exceptions.MlflowException as e:
    print(f"Failed to register model '{registry_model_name}': {str(e)}")

model_details
# Note: The variable `model_details` now contains details about the registered model

# COMMAND ----------

# MAGIC %md
# MAGIC ### Updating Model Metadata in the MLflow Model Registry
# MAGIC
# MAGIC In this step, we accomplish two primary tasks:
# MAGIC
# MAGIC 1. **Update Registered Model Metadata**: 
# MAGIC    - We attempt to update the description of an already registered model in the MLflow Model Registry. 
# MAGIC    - The description aims to clarify the purpose of the model, in this case, "This model predicts whether a bank customer will churn or not."
# MAGIC
# MAGIC 2. **Update Version-Specific Metadata**:
# MAGIC    - We update the metadata for a specific version of the model. 
# MAGIC    - Here, we add a description specifying that this model version is based on scikit-learn.
# MAGIC
# MAGIC Both operations are wrapped in try-except blocks for robust error handling. Should any operation fail, an error message will be printed to provide insight into the failure.
# MAGIC
# MAGIC > **Note**: The `model_details` variable is assumed to contain essential information about the registered model and its specific version.
# MAGIC

# COMMAND ----------

# Update the metadata of an already registered model
try:
    client.update_registered_model(
        name=model_details.name,
        description="This model predicts whether a bank customer will churn or not."
    )
    print(f"Successfully updated the description for the registered model '{model_details.name}'.")
except mlflow.exceptions.MlflowException as e:
    print(f"Failed to update the registered model '{model_details.name}': {str(e)}")

# Update the metadata for a specific version of the model
try:
    client.update_model_version(
        name=model_details.name,
        version=model_details.version,
        description="This is a scikit-learn based model."
    )
    print(f"Successfully updated the description for version {model_details.version} of the model '{model_details.name}'.")
except mlflow.exceptions.MlflowException as e:
    print(f"Failed to update version {model_details.version} of the model '{model_details.name}': {str(e)}")

# Note: The `model_details` variable is assumed to contain details about the registered model and its version

# COMMAND ----------

# MAGIC %md
# MAGIC ### Transitioning Model Version to 'Staging' Stage in the MLflow Model Registry
# MAGIC
# MAGIC In this step, the following objectives are met:
# MAGIC
# MAGIC 1. **Transition Model Version**:
# MAGIC    - We aim to transition a specific version of the registered model to the 'Staging' stage in the MLflow Model Registry.
# MAGIC   
# MAGIC 2. **Archiving Existing Versions**: 
# MAGIC    - The `archive_existing_versions=True` flag ensures that any pre-existing versions of the model in the 'Staging' stage are archived. This helps in keeping only the most relevant version in the stage.
# MAGIC
# MAGIC 3. **Error Handling**: 
# MAGIC    - The operation is wrapped in a try-except block. If the transition operation fails for any reason, a detailed error message will be displayed to help diagnose the issue.
# MAGIC
# MAGIC > **Note**: Successful completion will print a message confirming the successful transition of the model version to the 'Staging' stage.
# MAGIC
# MAGIC

# COMMAND ----------

# Transition the model version to the 'Staging' stage in the model registry
try:
    client.transition_model_version_stage(
        name=model_details.name,
        version=model_details.version,
        stage="Staging",
        archive_existing_versions=True  # Archives any existing versions in the 'Staging' stage
    )
    print(f"Successfully transitioned version {model_details.version} of the model '{model_details.name}' to 'Staging'.")
except mlflow.exceptions.MlflowException as e:
    print(f"Failed to transition version {model_details.version} of the model '{model_details.name}' to 'Staging': {str(e)}")


# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Registry Webhooks
# MAGIC
# MAGIC ### Supported Events
# MAGIC * **MODEL_VERSION_CREATED**: A new model version was created for the associated model.
# MAGIC * **MODEL_VERSION_TRANSITIONED_STAGE**: A model version’s stage was changed.
# MAGIC * **TRANSITION_REQUEST_CREATED**: A user requested a model version’s stage be transitioned.
# MAGIC * **COMMENT_CREATED**: A user wrote a comment on a registered model.
# MAGIC * **REGISTERED_MODEL_CREATED**: A new registered model was created. This event type can only be specified for a registry-wide webhook, which can be created by not specifying a model name in the create request.
# MAGIC * **MODEL_VERSION_TAG_SET**: A user set a tag on the model version.
# MAGIC * **MODEL_VERSION_TRANSITIONED_TO_STAGING**: A model version was transitioned to staging.
# MAGIC * **MODEL_VERSION_TRANSITIONED_TO_PRODUCTION**: A model version was transitioned to production.
# MAGIC * **MODEL_VERSION_TRANSITIONED_TO_ARCHIVED**: A model version was archived.
# MAGIC * **TRANSITION_REQUEST_TO_STAGING_CREATED**: A user requested a model version be transitioned to staging.
# MAGIC * **TRANSITION_REQUEST_TO_PRODUCTION_CREATED**: A user requested a model version be transitioned to production.
# MAGIC * **TRANSITION_REQUEST_TO_ARCHIVED_CREATED**: A user requested a model version be archived.
# MAGIC
# MAGIC ### Types of webhooks
# MAGIC * **HTTP webhook** &mdash; send triggers to endpoints of your choosing such as slack, AWS Lambda, Azure Functions, or GCP Cloud Functions
# MAGIC * **Job webhook** &mdash; trigger a job within the Databricks workspace

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow Endpoint Utility Functions
# MAGIC
# MAGIC This script contains utility functions to interact with MLflow REST API endpoints. The code imports necessary modules, initializes an MLflow client, and defines a series of functions to handle REST API calls. Below are the key components:
# MAGIC
# MAGIC ### Import Statements
# MAGIC
# MAGIC - `http_request from mlflow.utils.rest_utils`: Required for making HTTP requests to the MLflow server.
# MAGIC - `json`: Standard library for handling JSON formatted data.
# MAGIC
# MAGIC ### Functions
# MAGIC
# MAGIC #### `get_mlflow_client()`
# MAGIC - **Purpose**: Returns an initialized MLflowClient object for further operations.
# MAGIC - **Return Type**: `MlflowClient`
# MAGIC
# MAGIC #### `get_host_creds(client)`
# MAGIC - **Parameters**: `client` - Initialized MlflowClient object.
# MAGIC - **Purpose**: Fetches the host and token credentials from the MLflow tracking server.
# MAGIC - **Return Type**: Host and token credentials.
# MAGIC
# MAGIC #### `mlflow_call_endpoint(endpoint, method, body='{}')`
# MAGIC - **Parameters**:
# MAGIC   - `endpoint` (str): The MLflow API endpoint to call.
# MAGIC   - `method` (str): HTTP method to use ('GET' or other HTTP methods).
# MAGIC   - `body` (str, optional): JSON-formatted request payload, default is an empty JSON object.
# MAGIC - **Purpose**: Makes a REST API call to the specified MLflow endpoint.
# MAGIC - **Return Type**: Dictionary containing the JSON response from the API call or `None` if the request fails.
# MAGIC - **Error Handling**: Captures exceptions and prints an error message detailing the failure.
# MAGIC
# MAGIC ### Client Initialization and Credential Retrieval
# MAGIC
# MAGIC After defining the functions, the script initializes an `MlflowClient` object and fetches the host and token credentials.
# MAGIC
# MAGIC - `client = get_mlflow_client()`: Initializes the client.
# MAGIC - `host_creds = get_host_creds(client)`: Retrieves host and token credentials.
# MAGIC - `host = host_creds.host`: Extracts the host.
# MAGIC - `token = host_creds.token`: Extracts the token.
# MAGIC
# MAGIC

# COMMAND ----------

from mlflow.utils.rest_utils import http_request
import json

def get_mlflow_client():
    """Returns an initialized MLflowClient object."""
    return mlflow.tracking.client.MlflowClient()

def get_host_creds(client):
    """Fetches host and token credentials."""
    return client._tracking_client.store.get_host_creds()

def mlflow_call_endpoint(endpoint, method, body='{}'):
    """Calls an MLflow REST API endpoint.
    
    Parameters:
        endpoint (str): The endpoint to call.
        method (str): HTTP method ('GET' or other HTTP methods).
        body (str): JSON-formatted request payload.
        
    Returns:
        dict: JSON response as a dictionary.
    """
    host_creds = get_host_creds(get_mlflow_client())
    
    try:
        if method == 'GET':
            response = http_request(
                host_creds=host_creds,
                endpoint=f"/api/2.0/mlflow/{endpoint}",
                method=method,
                params=json.loads(body)
            )
        else:
            response = http_request(
                host_creds=host_creds,
                endpoint=f"/api/2.0/mlflow/{endpoint}",
                method=method,
                json=json.loads(body)
            )
        
        return response.json()
        
    except Exception as e:
        print(f"Failed to call MLflow endpoint '{endpoint}': {str(e)}")
        return None


client = get_mlflow_client()
host_creds = get_host_creds(client)
host = host_creds.host
token = host_creds.token

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setting Up Slack Notifications and Webhooks
# MAGIC
# MAGIC You can read more about Slack webhooks [here](https://api.slack.com/messaging/webhooks#create_a_webhook).
# MAGIC
# MAGIC First, we set up a webhook to notify us whenever a **New model version is created**.
# MAGIC
# MAGIC In the next cell assign the slack_webhook variable the link to your webhook. It should look as follows`"https://hooks.slack.com/services/?????????/??????????/????????????????????????"`

# COMMAND ----------

slack_webhook = "https://hooks.slack.com/services/?????????/??????????/???????????????????????"

# COMMAND ----------

import json 

trigger_for_slack = json.dumps({
  "model_name": registry_model_name,
  "events": ["MODEL_VERSION_CREATED"],
  "description": "Triggered when a new model version is created.",
  "http_url_spec": {
    "url": slack_webhook
  }
})
 
mlflow_call_endpoint("registry-webhooks/create", method = "POST", body = trigger_for_slack)

# COMMAND ----------

# MAGIC %md
# MAGIC Similarly we can create a webhook that notifies us when a **New transition request is made for a mode version**.

# COMMAND ----------

trigger_for_slack = json.dumps({
  "model_name": registry_model_name,
  "events": ["TRANSITION_REQUEST_CREATED"],
  "description": "Triggered when a new transition request for a model has been made.",
  "http_url_spec": {
    "url": slack_webhook
  }
})

mlflow_call_endpoint("registry-webhooks/create", method = "POST", body = trigger_for_slack)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Listing all webhooks.

# COMMAND ----------

list_model_webhooks = json.dumps({"model_name": registry_model_name})

model_webhooks = mlflow_call_endpoint("registry-webhooks/list", method = "GET", body = list_model_webhooks)
model_webhooks

# COMMAND ----------

# MAGIC %md
# MAGIC You can also **delete webhooks**.
# MAGIC
# MAGIC You can use the below cell to delete webhooks by ID or delete all the webhooks for a specific model.

# COMMAND ----------

# for webhook in model_webhooks["webhooks"]:
#     mlflow_call_endpoint(
#     "registry-webhooks/delete",
#     method="DELETE",
#     body=json.dumps({'id': webhook["id"]})
# )

# COMMAND ----------


