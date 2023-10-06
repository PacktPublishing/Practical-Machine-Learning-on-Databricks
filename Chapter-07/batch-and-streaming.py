# Databricks notebook source
# MAGIC %md 
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
# MAGIC ## Batch Deployment
# MAGIC
# MAGIC This notebook will go over the most common model deployment option of batch inferencing. We will load the latest model version for our <b>Bank customer churn prediction</b> problem from the model registry and load it as a python function that can be applied to a Spark Dataframe.

# COMMAND ----------

# MAGIC %md ### Inference in Spark
# MAGIC
# MAGIC Till now we have seen how you can use differnent machine learning libraries to train your model. When it comes to deployment we can now utilize to power of Spark to distribute our trained model to more than a single node and do predictions at scale.
# MAGIC
# MAGIC To do this, we will use `mlflow.pyfunc.spark_udf` and pass in the `SparkSession`, name of the model, and run id.
# MAGIC
# MAGIC <b>Note:</b> Using UDF's in Spark means that supporting libraries must be installed on every node in the cluster.  In the case of `sklearn`, this is installed in Databricks clusters by default.  When using other libraries, you will need to install them to ensure that they will work as UDFs.  

# COMMAND ----------

# MAGIC %md
# MAGIC First we will load the desired model from the model registry.

# COMMAND ----------

import mlflow

# the name of the model in the registry
registry_model_name = "Churn Prediction Bank"

# get the latest version of the model in staging and load it as a spark_udf.
# MLflow easily produces a Spark user defined function (UDF).  This bridges the gap between Python environments and applying models at scale using Spark.
model = mlflow.pyfunc.spark_udf(spark, model_uri=f"models:/{registry_model_name}/staging")

# COMMAND ----------

# MAGIC %md
# MAGIC This model was trained on raw dataset and using the Databricks AutoML. 
# MAGIC
# MAGIC <b>Note:</b> Make sure the dataset we want to run infrence on matches the schema of the dataset the model was trained on. In the current example we will simply reuse the dataset we used to train our model.
# MAGIC - As best practice keep all the model specific transformations like imputing missing values or scaling a column value should be done as part of the model pipelne and not when registering a table as feature table.

# COMMAND ----------

spark_df = spark.table("bank_churn_analysis.raw_Data")
display(spark_df)

# COMMAND ----------

# MAGIC %md
# MAGIC <b>Note:</b> we will not send RowNumber, CustomerId, Surname and Exited columns to the model.

# COMMAND ----------

exclude_colums = {'RowNumber', "CustomerId", "Surname", "Exited"}
input_columns = [col for col in spark_df.columns if col not in exclude_colums]
input_columns

# COMMAND ----------

# MAGIC %md Apply the model as a standard UDF using the column names as the input to the function.

# COMMAND ----------

#passing non label columns to the model as input
prediction_df = spark_df.withColumn("prediction", model(*input_columns))

display(prediction_df)

# COMMAND ----------

# MAGIC %md
# MAGIC <b>Now you can write the inference out to a database for fast access, to a Delta table, or any other file format depending on your application need.</b>

# COMMAND ----------

# MAGIC %md
# MAGIC __Note:__ In the above example we showcased how you can use mlflow API to perform batch inference. We didnt make use of the model trained on feature table that we created in Chapter 2. If you  want to utilize feature store API to log a trained model and also perform the batch inference check the notebook in Chapter 4 that has details on that.

# COMMAND ----------

# MAGIC %md # Streaming Deployment

# COMMAND ----------

# MAGIC %md
# MAGIC We can also perform continuous model inference using a technology like Spark's Structured Streaming. you can read more about this [here](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html). Using Spark for ingesting and building your Streaming ingestion pipelines and model insfrence solution is that:
# MAGIC - It offers the same Dataframe API to processing streaming data as you would use with batch data.
# MAGIC - provides a scalable and fault tolerant way to continuously perform inference on incoming new data.
# MAGIC
# MAGIC We will not go into detail of Spark structured streaming here but will cover how you can deploy model for inference on a stream of data.
# MAGIC
# MAGIC The first is usually to connect to a streaming data source like Kafka, Azure event bus or Kinesis. Using Spark structured streaming you can also simulate reading files as stream from a cloud storage like S3. For our example we are going to do just that.
# MAGIC
# MAGIC We'll read Delta table as a stream.

# COMMAND ----------

# right now we are just defining a streaming data source but this statement will not execute until we call an Spark action.
raw_streaming_df = spark.readStream.format("delta").option("ignoreChanges", "true").table("bank_churn_analysis.raw_Data").drop(*("RowNumber", "CustomerId", "Surname", "Exited"))

# if you want to read from a S3 location then use the next set of code
# streaming_data = (spark
#                  .readStream
#                  .schema(schema)
#                  .option("maxFilesPerTrigger", 1)
#                  .parquet("<location of parquet file>")
#                  .drop(*("RowNumber", "CustomerId", "Surname", "Exited")))

# COMMAND ----------

# we will use this to keep track of our streaming job
stream_name = "streaming_inference"

# COMMAND ----------

predictions_df = raw_streaming_df.withColumn("prediction", model(*raw_streaming_df.columns))
display(predictions_df, streamName=stream_name)

# COMMAND ----------

# Spark structured stream takes some time to finish initializing and trying to shut it off will throw an error if its not active. This code will prevent it.
active_streams = [stream.name for stream in spark.streams.active]
active_streams

import time
start_time = time.time()
while stream_name not in active_streams:
  time.sleep(5)
  # wait for 20 seconds to let the strem initialize
  if time.time()-start_time>20:
    # stream initialization was not kicked off or there is some network issue.
    break

# COMMAND ----------

# We will stop the stream after reviewing results
for stream in spark.streams.active:
    print(f"Stopping {stream.name}")
    stream.stop() # Stop the stream

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Write to Delta table

# COMMAND ----------

working_dir = "/tmp"
# this is important for streaming queries to keep track of what records have been processed and guyrantee each record is processed only once.
checkpoint_location = f"{working_dir}/stream.checkpoint"
# this is a temporary location where we will write the predictions of our model as Delta table
write_path = f"{working_dir}/predictions"

(predictions_df
    .writeStream                                           # Write the stream
    .queryName(stream_name)                                # Name the query
    .format("delta")                                       # Use the delta format
    .option("checkpointLocation", checkpoint_location)     # Specify where to log metadata
    .option("path", write_path)                            # Specify the output path
    .outputMode("append")                                  # "append" means append the new data to the table
    .start()                                               # Start the operation
)

# COMMAND ----------

# MAGIC %md
# MAGIC we can take a look at what files are written to the file system

# COMMAND ----------

# MAGIC %fs
# MAGIC ls /tmp/predictions/

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from delta.`/tmp/predictions`

# COMMAND ----------

# We will stop the stream after writing the data to the delta table
for stream in spark.streams.active:
    print(f"Stopping {stream.name}")
    stream.stop() # Stop the stream
