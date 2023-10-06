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
# MAGIC     - Click on `Create Cluster` and wait for your cluster to be provisioned.
# MAGIC
# MAGIC 2. **Attach this Notebook to Your Cluster**: 
# MAGIC     - Click on the menu labeled `Detached` at the top left of this workbook.
# MAGIC     - Select your cluster name to attach this notebook to your cluster.
# MAGIC --------
# MAGIC ### Outline
# MAGIC
# MAGIC We simulate a batch inference scenario where we train, deploy, and maintain a model to predict monthly Sales for ecommerce website on monthly basis. 
# MAGIC
# MAGIC **Data interval**: Arrives monthly <br> 
# MAGIC **Date range**: 01/01/2023 - 03/31/2023
# MAGIC
# MAGIC **Workflow**: 
# MAGIC * Load the new month of incoming data
# MAGIC * Apply incoming data checks 
# MAGIC   * Error and drift evaluation
# MAGIC * Identify and address any errors and drifts
# MAGIC * Train a new model
# MAGIC * Apply model validation checks versus the existing model in production
# MAGIC     * If checks pass, deploy the new candidate model to production
# MAGIC     * If checks fail, do not deploy the new candidate model <br>
# MAGIC     
# MAGIC **Reproducibility Tools**: 
# MAGIC * [MLflow](https://www.mlflow.org/docs/latest/index.html) for model parameters, metrics, and artifacts
# MAGIC * [Delta](https://docs.delta.io/latest/index.html) for data versioning <br>
# MAGIC
# MAGIC Although this notebook specifically addresses tests to monitor a supervised ML model for batch inference, the same tests are applicable in streaming and real-time settings.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Run setup and utils notebooks

# COMMAND ----------

# MAGIC %run ./config/setup

# COMMAND ----------

# MAGIC %run ./util/training

# COMMAND ----------

# MAGIC %run ./data/datagen

# COMMAND ----------

# MAGIC %run ./util/monitoring

# COMMAND ----------

# Remove all existing widgets
dbutils.widgets.removeAll()

# Create three widgets for the stats threshold limit, p-threshold, and min model R2 threshold
dbutils.widgets.text("stats_threshold_limit", "0.5")
dbutils.widgets.text("p_threshold", "0.05")
dbutils.widgets.text("min_model_r2_threshold", "0.005")

# Get the values of the widgets
# stats_threshold_limit: how much we should allow basic summary stats to shift
stats_threshold_limit = float(dbutils.widgets.get("stats_threshold_limit"))

# p_threshold: the p-value below which to reject null hypothesis
p_threshold = float(dbutils.widgets.get("p_threshold"))

# min_model_r2_threshold: minimum model improvement
min_model_r2_threshold = float(dbutils.widgets.get("min_model_r2_threshold"))

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Drift Dummy Dataset

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC ### Month 1 - Base line Data
# MAGIC
# MAGIC We have generated a dummy dataset to showcase model drift. The dataset consists of time series data for three months. The independent features of the dataset are:
# MAGIC
# MAGIC | Feature | Type | Description |
# MAGIC |---|---|---|
# MAGIC | Date | date | The date for which the record belongs. |
# MAGIC | Temperature | numeric | The highest daily temperature in Fahrenheit. |
# MAGIC | Weather_Condition | categorical | The weather condition, which can be sunny, cloudy, or rainy. |
# MAGIC | Promotion_Type | categorical | The type of promotion, which can be a discount, free gift, or bundle deal. |
# MAGIC | Website_Traffic | numeric | The total website traffic. |
# MAGIC | Device_Type | categorical | The type of device used to access the website, which can be mobile, desktop, or tablet. |
# MAGIC
# MAGIC The target variable of the dataset is Daily_Sales (numeric). Daily_Sales has the following correlations with the independent features for the first month:
# MAGIC
# MAGIC * Positive correlation with Temperature and Website_Traffic.
# MAGIC * Negative correlation with Weather_Condition and Device_Type.
# MAGIC
# MAGIC ### Data and Model Management
# MAGIC
# MAGIC #### Variables
# MAGIC
# MAGIC The following variables are also defined during our setup to help with execution down the line:
# MAGIC
# MAGIC Variable | Description
# MAGIC ---|---
# MAGIC `project_home_dir` | The path to the project home directory.
# MAGIC `raw_good_data_path` | The path to the directory where the raw data is stored as csv.
# MAGIC `raw_month2_bad_data_path` | The path to the directory where the bad data for simulating feature drift is stored as csv.
# MAGIC `months_gold_path` | The path to the directory where the clean and processed data is stored in Delta format.
# MAGIC `mlflow_experiment_name` | The name of the MLflow experiment where the model will be registered.
# MAGIC `mlflow_experiment_path` | The path relative to our home directory in the workspace where the experiment will be located.
# MAGIC
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

print(f'good raw data file location : {raw_good_data_path}')
print(f'bad raw data location : {raw_month2_bad_data_path}')
print(f'Gold Delta table path : {months_gold_path}')
print(f'MLflow experiment name : {mlflow_experiment_name}')
print(f'MLflow experiment path : {mlflow_experiment_path}')

# COMMAND ----------

# MAGIC %md
# MAGIC #### i. Initial Data load
# MAGIC
# MAGIC Load the first month of data which we use to train and evaluate our first model.
# MAGIC
# MAGIC We create a "Gold" table to which we will be appending each subsequent month of data.
# MAGIC

# COMMAND ----------

# Ensure we start with no existing Delta table 
dbutils.fs.rm(months_gold_path, True)

# Incoming Month 1 Data
raw_data = spark.read.csv(raw_good_data_path, header=True, inferSchema=True)

# Filter the DataFrame to only include data for January 2023
raw_data_month1 = raw_data.filter(raw_data["Date"].between("2023-01-01", "2023-01-31"))

# Print the filtered DataFrame
raw_data_month1.show()

# COMMAND ----------

import pyspark.sql.functions as F
# Create inital version of the Gold Delta table we will use for training - this will be updated with subsequent "months" of data
raw_data_month1.withColumn("month", F.lit("month_1")).write.format("delta").mode("overwrite").partitionBy("month").save(months_gold_path)

# COMMAND ----------

#list files in the gold delta table path
display(dbutils.fs.ls(months_gold_path))

# COMMAND ----------

# MAGIC %md
# MAGIC #### ii. Model Training

# COMMAND ----------

#read gold data for month 1 from the Delta table
month1_gold_delta_table = DeltaTable.forPath(spark, path=months_gold_path)
month1_gold_df = month1_gold_delta_table.toDF()

# Set the month number - used for naming the MLflow run and tracked as a parameter 
month = 1

# Specify name of MLflow run
run_name = f"month_{month}"

target_col = "Daily_Sales"
cat_cols = [col[0] for col in month1_gold_df.dtypes if col[1]=="string" and col[0]!='month']
num_cols= [col[0] for col in month1_gold_df.dtypes if ((col[1]=="int" or col[1]=="double") and col[0]!="Daily_Sales") ]

print(f"category columns : {cat_cols}")
print(f"numeric columns : {num_cols}")
print(f"target column : {target_col}")

# Define the parameters to pass in the RandomForestRegressor model
model_params = {"n_estimators": 500,
                "max_depth": 5,
                "max_features": "log2"}

# Define a dictionary of parameters that we would like to use during preprocessing
misc_params = {"month": month,
               "target_col": target_col,
               "cat_cols": cat_cols,
               "num_cols": num_cols}             

# COMMAND ----------

# Trigger model training and logging to MLflow
month1_run = train_sklearn_rf_model(run_name, 
                        months_gold_path, 
                        model_params, 
                        misc_params)


month_1_run_id = month1_run.info.run_id                        

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### iii. Model Deployment
# MAGIC We first register the model to the MLflow Model Registry. For demonstration purposes, we will immediately transition the model to the "Production" stage in the MLflow Model Registry. However, in a real-world scenario, one should have a robust model validation process in place prior to migrating a model to Production.
# MAGIC
# MAGIC We will demonstrate a multi-stage approach in the subsequent sections:
# MAGIC 1. Transitioning the model to the "Staging" stage.
# MAGIC 2. Conducting model validation checks.
# MAGIC 3. Only then, triggering a transition from Staging to Production once these checks are satisfied.
# MAGIC
# MAGIC

# COMMAND ----------

# Register model to MLflow Model Registry
month_1_model_version = mlflow.register_model(model_uri=f"runs:/{month_1_run_id}/model", name=mlflow_experiment_name)

# COMMAND ----------

# Transition model to Production
month_1_model_version = transition_model(month_1_model_version, stage="Production")
print(month_1_model_version)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Month 2 - Arrival of New Data
# MAGIC
# MAGIC After deploying our model for a month, we are now faced with the arrival of a fresh month's worth of data. Let's explore two scenarios related to this new data:
# MAGIC
# MAGIC **Scenario 1: Missing values in website_traffic**
# MAGIC An updated upstream Data cleaning process has a bug causing the the value of, `website_traffic` counts for promotion type `bundle_deal` and `free_gift` to be empty.  
# MAGIC
# MAGIC **Scenario 2: Introduction of new measurement for temperature**
# MAGIC Also during the upstream data generation procedure a the temperature values are now being captured in __Celcius__ rather than in __Fahrenheit__.
# MAGIC
# MAGIC **What are we simulating here?**
# MAGIC In this scenario, we are simulating two important factors:
# MAGIC - Feature drift: The characteristics of the data have changed over time, specifically with missing `website_traffic` entries for `bundle_deal` and `free_gift`.
# MAGIC - Upstream data errors: Unexpected changes or additions in the data generation process, such as the introduction of a different unit of measuring temperature.

# COMMAND ----------

# MAGIC %md
# MAGIC #### i. Feature checks prior to model training
# MAGIC
# MAGIC **All features**
# MAGIC * Null checks
# MAGIC
# MAGIC **Numeric features**
# MAGIC * Summary statistic checks: mean, median, standard deviation, minimum, maximum
# MAGIC * Distribution checks
# MAGIC
# MAGIC **Categorical features**
# MAGIC * Check expected count for each level
# MAGIC * Check the mode

# COMMAND ----------


# Incoming Month 2 Data
raw_data_month2 = spark.read.csv(raw_month2_bad_data_path, header=True, inferSchema=True)

# Filter the DataFrame to only include data for Feb 2023
raw_data_month2 = raw_data_month2.filter(raw_data_month2["Date"].between("2023-02-01", "2023-02-28"))

# Print the filtered DataFrame
raw_data_month2.show(5)

# COMMAND ----------

# Compute summary statistics on new incoming data
# we will keep only the columns that we monitored for the last mode training data
# convert to pandas dataframe should be used with care as if the size of data is larger than what can fit on driver node then this can cause failures.
# In the case of data size being large use proper sampling technique to estimate population summary statistics.
month_2_pdf = raw_data_month2.toPandas().drop(['Date'], axis=1)
summary_stats_month_2_pdf = calculate_summary_stats(month_2_pdf)
summary_stats_month_2_pdf

# COMMAND ----------

# Get the original MLflow run associated with the model registered under Production
current_prod_run = get_run_from_registered_model(mlflow_experiment_name, stage="Production")

# Load in original versions of Delta table used at training time for current Production model
current_prod_pdf = load_delta_table_from_run(current_prod_run).toPandas()

# Load summary statistics pandas DataFrame for data which the model currently in Production was trained and evaluated against
current_prod_stats_pdf = load_summary_stats_pdf_from_run(current_prod_run, project_local_tmp_dir)
current_prod_stats_pdf

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **All features**
# MAGIC * Null checks

# COMMAND ----------

print("\nCHECKING PROPORTION OF NULLS.....")
check_null_proportion(month_2_pdf, null_proportion_threshold=.5)

# COMMAND ----------

# MAGIC %md
# MAGIC **Numeric features**
# MAGIC * Summary statistic checks: mean, median, standard deviation, minimum, maximum
# MAGIC * Distribution checks

# COMMAND ----------

statistic_list = ["mean", "median", "std", "min", "max"]

unique_feature_diff_array_month_2 = check_diff_in_summary_stats(summary_stats_month_2_pdf, 
                                                                current_prod_stats_pdf, 
                                                                num_cols + [target_col], 
                                                                stats_threshold_limit, 
                                                                statistic_list)

unique_feature_diff_array_month_2

# COMMAND ----------

print(f"Let's look at the box plots of the features that exceed the stats_threshold_limit of {stats_threshold_limit}")
plot_boxplots(unique_feature_diff_array_month_2, current_prod_pdf, month_2_pdf)

# COMMAND ----------

print("\nCHECKING VARIANCES WITH LEVENE TEST.....")
check_diff_in_variances(current_prod_pdf, month_2_pdf, num_cols, p_threshold)

print("\nCHECKING KS TEST.....")
check_dist_ks_bonferroni_test(current_prod_pdf, month_2_pdf, num_cols + [target_col], p_threshold)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Categorical features**
# MAGIC * Check expected count for each level
# MAGIC * Check the mode

# COMMAND ----------

check_categorical_diffs(current_prod_pdf, month_2_pdf, cat_cols, p_threshold)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **`Action`: Resolve Data issues**
# MAGIC
# MAGIC After identifying data issues with `Temperature` and `Website_Traffic` and collaborating with the upstream data processing team, we have successfully resolved these issues. The fixed data for the new month is incorporated into our Gold Delta table. Subsequently, we proceed with training on the updated dataset to leverage the newly available information.

# COMMAND ----------

# Incoming corrected Data
raw_data = spark.read.csv(raw_good_data_path, header=True, inferSchema=True)

# Filter the DataFrame to only include data for January 2023
raw_data_month2 = raw_data.filter(raw_data["Date"].between("2023-02-01", "2023-02-28"))

# Append new month of data to Gold Delta table to use for training
raw_data_month2.withColumn("month", F.lit("month_2")).write.format("delta").partitionBy("month").mode("append").save(months_gold_path)

# COMMAND ----------

# MAGIC %md
# MAGIC #### ii. Model Training
# MAGIC
# MAGIC Retrain the same model, but this time we are able to use an extra month of data

# COMMAND ----------

# Set the month number - used for naming the MLflow run and tracked as a parameter 
month = 2

# Specify name of MLflow run
run_name = f"month_{month}"

# Define the parameters to pass in the RandomForestRegressor model
model_params = {"n_estimators": 500,
                "max_depth": 5,
                "max_features": "log2"}

# Define a dictionary of parameters that we would like to use during preprocessing
misc_params = {"month": month,
               "target_col": target_col,
               "cat_cols": cat_cols,
               "num_cols": num_cols}

# COMMAND ----------

# Trigger model training and logging to MLflow
month2_run = train_sklearn_rf_model(run_name, 
                        months_gold_path, 
                        model_params, 
                        misc_params)


month_2_run_id = month2_run.info.run_id        

# COMMAND ----------

# Register model to MLflow Model Registry
month_2_model_version = mlflow.register_model(model_uri=f"runs:/{month_2_run_id}/model", name=mlflow_experiment_name)

# Transition model to Staging
month_2_model_version = transition_model(month_2_model_version, stage="Staging")
print(month_2_model_version)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### iii. Model checks prior to model deployment

# COMMAND ----------

# Get the original MLflow run associated with the model registered under Staging
current_staging_run = get_run_from_registered_model(mlflow_experiment_name, stage="Staging")

metric_to_check = "r2_score_X_test"
compare_model_perfs(current_staging_run, current_prod_run, min_model_r2_threshold, metric_to_check)

# COMMAND ----------

month_2_model_version = transition_model(month_2_model_version, stage="Production")
print(month_2_model_version)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Month 3 - New Data Arrives
# MAGIC
# MAGIC We have had a model in production for 2 months now and have now obtained an additional month of data.
# MAGIC
# MAGIC **Scenario 2:**
# MAGIC * A product campaign went viral on social media. Sales increased by 30% for each day.
# MAGIC
# MAGIC **What are we simulating here?**
# MAGIC * Label drift
# MAGIC * Concept drift
# MAGIC   * The underlying relationship between the features and label has changed due to a viral marketing campaign.

# COMMAND ----------

# MAGIC %md
# MAGIC #### i. Feature checks prior to model training
# MAGIC
# MAGIC **All features**
# MAGIC * Null checks
# MAGIC
# MAGIC **Numeric features**
# MAGIC * Summary statistic checks: mean, median, standard deviation, minimum, maximum
# MAGIC * Distribution checks
# MAGIC
# MAGIC **Categorical features**
# MAGIC * Check expected count for each level
# MAGIC * Check the mode

# COMMAND ----------

# Incoming Month 1 Data
raw_data = spark.read.csv(raw_good_data_path, header=True, inferSchema=True)

# Filter the DataFrame to only include data for January 2023
raw_data_month3 = raw_data.filter(raw_data["Date"].between("2023-03-01", "2023-03-31"))

# Print the filtered DataFrame
raw_data_month3.show(5)

# COMMAND ----------

# Compute summary statistics on new incoming data
# we will keep only the columns that we monitored for the last mode training data
# convert to pandas dataframe should be used with care as if the size of data is larger than what can fit on driver node then this can cause failures.
# In the case of data size being large use proper sampling technique to estimate population summary statistics.
month_3_pdf = raw_data_month3.toPandas().drop(['Date'], axis=1)
summary_stats_month_3_pdf = calculate_summary_stats(month_3_pdf)
summary_stats_month_3_pdf

# COMMAND ----------

# Get the current MLflow run associated with the model registered under Production
current_prod_run_2 = get_run_from_registered_model(mlflow_experiment_name, stage="Production")

# Load in original versions of Delta table used at training time for current Production model
current_prod_pdf_2 = load_delta_table_from_run(current_prod_run_2).toPandas()

# Load summary statistics pandas DataFrame for data which the model currently in Production was trained and evaluated against
current_prod_stats_pdf_2 = load_summary_stats_pdf_from_run(current_prod_run_2, project_local_tmp_dir)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **All features**
# MAGIC * Null checks

# COMMAND ----------

print("\nCHECKING PROPORTION OF NULLS.....")
check_null_proportion(month_3_pdf, null_proportion_threshold=.5)

# COMMAND ----------

# MAGIC %md
# MAGIC **Numeric features**
# MAGIC * Summary statistic checks: mean, median, standard deviation, minimum, maximum
# MAGIC * Distribution checks

# COMMAND ----------

unique_feature_diff_array_month_3 = check_diff_in_summary_stats(summary_stats_month_3_pdf, 
                                                                current_prod_stats_pdf_2, 
                                                                num_cols + [target_col], 
                                                                stats_threshold_limit, 
                                                                statistic_list)

unique_feature_diff_array_month_3

# COMMAND ----------

print(f"Let's look at the box plots of the features that exceed the stats_threshold_limit of {stats_threshold_limit}")
plot_boxplots(unique_feature_diff_array_month_3, current_prod_pdf_2, month_3_pdf)

# COMMAND ----------

print("\nCHECKING VARIANCES WITH LEVENE TEST.....")
check_diff_in_variances(current_prod_pdf_2, month_3_pdf, num_cols, p_threshold)

print("\nCHECKING KS TEST.....")
check_dist_ks_bonferroni_test(current_prod_pdf_2, month_3_pdf, num_cols + [target_col], p_threshold)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Categorical features**
# MAGIC * Check expected count for each level
# MAGIC * Check the mode

# COMMAND ----------

check_categorical_diffs(current_prod_pdf_2, month_3_pdf, cat_cols, p_threshold)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **`Action`: Include new data with label drift in training**
# MAGIC
# MAGIC We observe that our label has drifted, and after analysis observe that this most recent month of data was captured during a spike in sales caused by a viral marketing campaign. As such, we will retrain our model and include this recent month of data during training.

# COMMAND ----------

# Append the new month of data (where listings are most expensive across the board)
raw_data_month3.withColumn("month", F.lit("month_3")).write.format("delta").partitionBy("month").mode("append").save(months_gold_path)

# COMMAND ----------

# MAGIC %md
# MAGIC #### ii. Model Training
# MAGIC
# MAGIC Retrain the same model from previous months, including the additional month of data where the label has drifted.

# COMMAND ----------

# Set the month number - used for naming the MLflow run and tracked as a parameter 
month = 3

# Specify name of MLflow run
run_name = f"month_{month}"

# Define the parameters to pass in the RandomForestRegressor model
model_params = {"n_estimators": 500,
                "max_depth": 5,
                "max_features": "log2"}

# Define a dictionary of parameters that we would like to use during preprocessing
misc_params = {"month": month,
               "target_col": target_col,
               "cat_cols": cat_cols,
               "num_cols": num_cols}

# COMMAND ----------

# Trigger model training and logging to MLflow
month3_run = train_sklearn_rf_model(run_name, 
                        months_gold_path, 
                        model_params, 
                        misc_params)


month_3_run_id = month3_run.info.run_id     

# COMMAND ----------

# Register model to MLflow Model Registry
month_3_model_version = mlflow.register_model(model_uri=f"runs:/{month_3_run_id}/model", name=mlflow_experiment_name)

# Transition model to Staging
month_3_model_version = transition_model(month_3_model_version, stage="Staging")
print(month_3_model_version)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### iii. Model checks prior to model deployment

# COMMAND ----------

# Get the MLflow run associated with the model currently registered in Staging
current_staging_run_2 = get_run_from_registered_model(mlflow_experiment_name, stage="Staging")

metric_to_check = "r2_score_X_test"
compare_model_perfs(current_staging_run_2, current_prod_run_2, min_model_r2_threshold, metric_to_check)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC In this case we note that the new candidate model in Staging performs notably worse than the current model in Production. We know from our checks prior to training that the label has drifted, and that this was due to new listing prices being recorded during vacation season. At this point we would want to prevent a migration of the new candidate model directly to Production and instead investigate if there is any way we can improve model performance. This could involve tuning the hyperparameters of our model, or additionally investigating the inclusion of additional features such as "month of the year" which could allow us to capture temporal impacts to listing prices.
