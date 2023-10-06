# Databricks notebook source
# MAGIC %md # 

# COMMAND ----------

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
# MAGIC
# MAGIC ## Real Time Deployment Options
# MAGIC
# MAGIC * **Databricks Integrated Serving Endpoints**: These endpoints offer a comprehensive solution for both prototyping and production deployment of models. They are designed to manage real-time requests through REST APIs. We are going to cover this approach in the notebook.
# MAGIC
# MAGIC ### Additional options
# MAGIC
# MAGIC  MLflow integrates seamlessly with managed services across various cloud platforms if your intent is to use cloud specific model serving capabilities:
# MAGIC
# MAGIC - **Azure ML**: For Microsoft Azure
# MAGIC - **SageMaker**: For AWS
# MAGIC - **Vertex AI**: For Google Cloud Platform
# MAGIC
# MAGIC ### Custom Deployments
# MAGIC
# MAGIC If you're seeking a more custom deployment, you can:
# MAGIC
# MAGIC - Export the model from the Model Registry as a Python pickle file.
# MAGIC - Create your own Flask application to serve the model.
# MAGIC   
# MAGIC **Note**: This custom approach often leverages containerization technologies like Docker or orchestration solutions like Kubernetes.
# MAGIC
# MAGIC   

# COMMAND ----------

# MAGIC %md
# MAGIC ### Databricks Serving Endpoint
# MAGIC We will use the model for our Bank Customer Churn prediction that we enabled serving for through the UI. On the serving page you can find code snippets that show you exactly how to call the deployed model. Here we are going to dynamically generate the URI for the deployed model so that you can execute this code in your workspace without change.

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

# MAGIC %md Defining a function called `score_model` that will pass JSON string as input to the model and get response back.

# COMMAND ----------

# Import the requests library for HTTP communication
import requests

#change the model_serving_endpoint_name to the one you have given.
model_serving_endpoint_name = "churn_prediction"

# Define the function 'score_model' which takes a dictionary as an input
def score_model(data_json: str):
    
    # Construct the URL for the model serving endpoint
    url = f"https://{instance}/serving-endpoints/{model_serving_endpoint_name}/invocations"
    
    # Make an HTTP POST request to score the model
    response = requests.request(method="POST", headers=headers, url=url, data=data_json)
    
    # Check if the request was successful (HTTP status code 200)
    if response.status_code != 200:
        # If not, raise an exception detailing the failure
        raise Exception(f"Request failed with status {response.status_code}, {response.text}")
        
    # Return the JSON response from the model scoring endpoint
    return response.json()

# COMMAND ----------

#reading a sample of raw data
raw_data_spark_df =  spark.table("bank_churn_analysis.raw_data")

input_cols = [col for col in raw_data_spark_df.columns if col not in {'RowNumber', 'CustomerId', 'Surname', 'Exited'}]

#drop the columns that will not be send to model as input
raw_data_spark_df = raw_data_spark_df.select(*[input_cols])

pandas_df = raw_data_spark_df.toPandas()
#convert to pandas dataframe

#lets take 2 sample records to use as input for our serving endpoint
input_examples_df_records = pandas_df[:2]
input_examples_df_records

# COMMAND ----------

# MAGIC %md
# MAGIC ### DataFrame Records Format
# MAGIC ####Overview
# MAGIC The DataFrame Records format is useful when the data can be readily represented as a Pandas DataFrame. In this approach, the DataFrame is serialized into a list of dictionaries, with each dictionary corresponding to a row in the DataFrame.
# MAGIC
# MAGIC ####Pros and Cons
# MAGIC - __Pros__: This format is easier to read and is more human-friendly.
# MAGIC - __Cons__: It consumes more bandwidth because the column names are repeated for each record.
# MAGIC
# MAGIC #### Use Case
# MAGIC This format is preferable when you need to send DataFrame-like data, and readability is a priority.

# COMMAND ----------

# Serialize using json
import json
serialized_data = json.dumps({"dataframe_records": input_examples_df_records.to_dict('records')}, indent=4)
print(serialized_data)
score_model(serialized_data)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### DataFrame Split Format
# MAGIC
# MAGIC #### Overview
# MAGIC
# MAGIC This format represents a Pandas DataFrame in a split orientation, separating the columns, index, and data into different keys. This is a more bandwidth-efficient alternative to the records orientation.
# MAGIC
# MAGIC #### Pros and Cons
# MAGIC
# MAGIC - __Pros__: This format is more bandwidth-efficient as compared to the records orientation.
# MAGIC - __Cons__: It is a bit less intuitive to read.
# MAGIC
# MAGIC #### Use Case
# MAGIC
# MAGIC This format is useful when sending DataFrame-like data, and bandwidth or payload size is a concern.

# COMMAND ----------

serialized_data = json.dumps({"dataframe_split": input_examples_df_records.to_dict('split')}, indent=4)
print(serialized_data)
score_model(serialized_data)
