# Databricks notebook source
# MAGIC %md
# MAGIC * [**Customer Churn**](https://en.wikipedia.org/wiki/Customer_attrition) also known as Customer attrition, customer turnover, or customer defection, is the loss of clients or customers and is...  
# MAGIC   * Built on top of Databricks Platform
# MAGIC   * Uses Databricks ML runtime and Feature store
# MAGIC * This Notebook...  
# MAGIC   * We will use Customer Churn dataset from the [Kaggle](https://www.kaggle.com/mathchi/churn-for-bank-customers).
# MAGIC   * We will skip the EDA part and focus on the feature engineering part and registering feature tables into Databricks feature store.

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
# MAGIC   - On `AWS`, select `i3.xlarge` / on `Azure`, select `Standard_DS4_V2` as __Node type__.
# MAGIC   - Click the `create cluster` button and wait for your cluster to be provisioned
# MAGIC 3. **Attach** this notebook to your cluster by...   
# MAGIC   - Click on your cluster name in menu `Detached` at the top left of this workbook to attach it to this workbook 

# COMMAND ----------

# MAGIC %md 
# MAGIC ###Step1: Ingest Data to Notebook
# MAGIC
# MAGIC We will download the dataset hosted at  [**Kaggle**](https://www.kaggle.com/mathchi/churn-for-bank-customers)
# MAGIC
# MAGIC ## Content
# MAGIC *   `RowNumber` —corresponds to the record (row) number and has no effect on the output.
# MAGIC *   `CustomerId` -contains random values and has no effect on customer leaving the bank.
# MAGIC *   `Surname` —the surname of a customer has no impact on their decision to leave the bank.
# MAGIC *   `CreditScore` —can have an effect on customer churn, since a customer with a higher credit score is less likely to leave the bank.
# MAGIC *   `Geography` —a customer’s location can affect their decision to leave the bank.
# MAGIC *   `Gender` —it’s interesting to explore whether gender plays a role in a customer leaving the bank
# MAGIC *   `Age` —this is certainly relevant, since older customers are less likely to leave their bank than younger ones.
# MAGIC *   `Tenure` —refers to the number of years that the customer has been a client of the bank. Normally, older clients are more loyal and less likely to leave a bank
# MAGIC *   `Balance` —also a very good indicator of customer churn, as people with a higher balance in their accounts are less likely to leave the bank compared to those with lower balances.
# MAGIC *   `NumOfProducts` —refers to the number of products that a customer has purchased through the bank.
# MAGIC *   `HasCrCard` —denotes whether or not a customer has a credit card. This column is also relevant, since people with a credit card are less likely to leave the bank.
# MAGIC *   `IsActiveMember` —active customers are less likely to leave the bank
# MAGIC *   `EstimatedSalary` —as with balance, people with lower salaries are more likely to leave the bank compared to those with higher salaries.
# MAGIC *   `Exited` —whether or not the customer left the bank.
# MAGIC
# MAGIC ## Acknowledgements
# MAGIC
# MAGIC As we know, it is much more expensive to sign in a new client than keeping an existing one.
# MAGIC It is advantageous for banks to know what leads a client towards the decision to leave the company.
# MAGIC Churn prevention allows companies to develop loyalty programs and retention campaigns to keep as many customers as possible.
# MAGIC
# MAGIC Data= https://www.kaggle.com/mathchi/churn-for-bank-customers 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Data
# MAGIC
# MAGIC Next, we'll import our data for this part

# COMMAND ----------

#read more about reading files from Databricks repos at https://docs.databricks.com/repos.html#access-files-in-a-repo-programmatically
import os
bank_df = spark.read.option("header", True).option("inferSchema", True).csv(f"file:{os.getcwd()}/data/churn.csv")
display(bank_df)

# COMMAND ----------

# MAGIC %md
# MAGIC We can drop RowNumber in the feature engineering step as this is not adding any valuable information.
# MAGIC
# MAGIC **Note:**  
# MAGIC Databricks introduced a built in data profiler for spark dataframes. The built in function display now gives an option to profile data automatically

# COMMAND ----------

display(bank_df)

# COMMAND ----------

# MAGIC %md Lets get unique value count in Surname

# COMMAND ----------

bank_df.select('Surname').distinct().count()

# COMMAND ----------

# MAGIC %md
# MAGIC As we can see Surname column have a lot of unique values and is not adding any useful information for us so we will drop it in our feature engineering step.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Feature Table
# MAGIC
# MAGIC Next, we can use the DataFrame **`bank_df`** to create a feature table using Feature Store.
# MAGIC
# MAGIC **In order to write our features out as a feature table we will perform the following steps:**
# MAGIC 1. Create a Database that will store any feature table. In our case let that be `bank_churn_analysis`
# MAGIC 1. Write the Python functions to compute the features. The output of each function should be an Apache Spark DataFrame with a unique primary key. The primary key can consist of one or more columns.
# MAGIC 1. Create a feature table by instantiating a FeatureStoreClient and using create_table (Databricks Runtime 10.2 ML or above) or create_feature_table (Databricks Runtime 10.1 ML or below).
# MAGIC 1. Populate the feature table using write_table.
# MAGIC
# MAGIC Note: 
# MAGIC - **If you want to prevent any data leakage you would want to consider not performing OHE or any feature treatment at the time of registering dataset as a feature table. **

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Defining a database to store feature tables.

# COMMAND ----------

DATABASE_NAME = "bank_churn_analysis"
#setup database that will hold our Feature tables in Delta format.
spark.sql(f"CREATE DATABASE IF NOT EXISTS {DATABASE_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC write the raw data out as a delta table

# COMMAND ----------

bank_df.write.format("delta").mode("overwrite").saveAsTable(f"{DATABASE_NAME}.raw_data")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Defining a feature engineering function that will return a Spark dataframe with a unique primary key. 
# MAGIC In our case it is the `CustomerId`.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC The `bank_df` DataFrame is already pretty clean, but we do have some nominal features that we'll need to convert to numeric features for modeling.
# MAGIC
# MAGIC These features include:
# MAGIC
# MAGIC * **`Geography`**
# MAGIC * **`Gender`**
# MAGIC
# MAGIC We will also be dropping few features which dont add additional value for our model:
# MAGIC * **`RowNumber`**
# MAGIC * **`Surname`**
# MAGIC
# MAGIC ### Create `compute_features` Function
# MAGIC
# MAGIC A lot of data scientists are familiar with Pandas DataFrames, so we'll use the [pyspark.pandas](https://spark.apache.org/docs/3.2.0/api/python/user_guide/pandas_on_spark/) library to one-hot encode these categorical features.
# MAGIC
# MAGIC **Note:** we are creating a function to perform these computations. We'll use it to refer to this set of instructions when creating our feature table.

# COMMAND ----------


import pyspark.pandas as ps
import numpy as np

def compute_features(spark_df):
    # https://spark.apache.org/docs/latest/api/python/migration_guide/koalas_to_pyspark.html?highlight=dataframe%20pandas_api
    # Convert to pyspark.pandas DataFrame
    ps_df = spark_df.pandas_api()
    
    # Drop RowNumber & Surname column
    ps_df = ps_df.drop(['RowNumber', 'Surname'], axis=1)
    
    # One-Hot Encoding for Geography and Gender
    ohe_ps_df = ps.get_dummies(
      ps_df, 
      columns=["Geography", "Gender"],
      dtype="int",
      drop_first=True
    )
    
    # Clean up column names
    ohe_ps_df.columns = ohe_ps_df.columns.str.replace(r' ', '', regex=True)
    ohe_ps_df.columns = ohe_ps_df.columns.str.replace(r'(', '-', regex=True)
    ohe_ps_df.columns = ohe_ps_df.columns.str.replace(r')', '', regex=True)
    
    ## Additional example feature engineering steps

    # # Create a binary feature indicating whether the balance is zero or not
    # ohe_ps_df['Is_Balance_Zero'] = (ohe_ps_df['Balance'] == 0).astype('int')
    
    # # Ratio of Tenure to Age
    # ohe_ps_df['Tenure_to_Age'] = ohe_ps_df['Tenure'] / ohe_ps_df['Age']
    
    # # Interaction feature: Balance to EstimatedSalary ratio
    # ohe_ps_df['Balance_to_Salary'] = ohe_ps_df['Balance'] / ohe_ps_df['EstimatedSalary']
    
    return ohe_ps_df


# COMMAND ----------

# MAGIC %md
# MAGIC ### Compute Features
# MAGIC
# MAGIC Next, we can use our featurization function `compute_features` to create create a DataFrame of our features.

# COMMAND ----------

bank_features_df = compute_features(bank_df)
display(bank_features_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ##3. Create the Feature Table
# MAGIC
# MAGIC Next, we can use the `feature_table` operation to register the DataFrame as a Feature Store table.
# MAGIC
# MAGIC In order to do this, we'll want the following details:
# MAGIC
# MAGIC 1. The `name` of the database and table where we want to store the feature table
# MAGIC 1. The `keys` for the table
# MAGIC 1. The `schema` of the table
# MAGIC 1. A `description` of the contents of the feature table
# MAGIC 1. `partition_columns`- Column(s) used to partition the feature table.
# MAGIC 1. `features_df`(optional) - Data to insert into this feature table. The schema of features_df will be used as the feature table schema.
# MAGIC
# MAGIC **Note:** 
# MAGIC 1. This creates our feature table, but we still need to write our values in the DataFrame to the table. 

# COMMAND ----------

#Our first step is to instantiate the feature store client using `FeatureStoreClient()`.
from databricks.feature_store import FeatureStoreClient
fs = FeatureStoreClient()

# COMMAND ----------

# MAGIC %md
# MAGIC We have __2__ options to initialize a feature table.
# MAGIC
# MAGIC 1. Providing Dataframe to populate feature table at time of defining feature table. This approach can be used when you have a feature dataframe ready to instantiate a feature table.
# MAGIC ``` 
# MAGIC bank_feature_table = fs.create_table(
# MAGIC   name=f"{DATABASE_NAME}.bank_customer_features", # the name of the feature table
# MAGIC   primary_keys=["CustomerId"], # primary key that will be used to perform joins
# MAGIC   schema=bank_features_df.spark.schema(), # the schema of the Feature table
# MAGIC   description="This customer level table contains one-hot encoded categorical and scaled numeric features to predict bank customer churn.",
# MAGIC   feature_df=bank_features_df.to_spark() 
# MAGIC )
# MAGIC ```
# MAGIC 2. In second case you can provide definition of the feature table without providing a source dataframe. This approach can be used when your data to populate feature store will be ingested at a different time then when you are defining the feature table. We will be showcasing this approach as part of the notebook.

# COMMAND ----------

bank_feature_table = fs.create_table(
  name=f"{DATABASE_NAME}.bank_customer_features", # the name of the feature table
  primary_keys=["CustomerId"], # primary key that will be used to perform joins
  schema=bank_features_df.spark.schema(), # the schema of the Feature table
  description="This customer level table contains one-hot encoded categorical and scaled numeric features to predict bank customer churn."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Populate the feature table using write_table.
# MAGIC Now, we can write the records from **`bank_features_df`** to the feature table.

# COMMAND ----------

fs.write_table(df=bank_features_df.to_spark(), name=f"{DATABASE_NAME}.bank_customer_features", mode="overwrite")
#instead of overwrite you can choose "merge" as an option if you want to update only certain records.

# COMMAND ----------

# MAGIC %md
# MAGIC ##5. Browsing the Feature Store
# MAGIC
# MAGIC The tables are now visible and searchable in the [Feature Store](/#feature-store/feature-store)

# COMMAND ----------

# MAGIC %md
# MAGIC Optionally if your usecase requires joining features for real time inference, you can write your features out to an [online store](https://docs.databricks.com/applications/machine-learning/feature-store.html#publish-features-to-an-online-feature-store).
# MAGIC
# MAGIC And finally, we can perform Access Control using built-in features in the Feature Store UI.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cleanup

# COMMAND ----------

#Drop feature table. This will drop the underlying Delta table as well.

# fs.drop_table(
#   name=f"{DATABASE_NAME}.bank_customer_features"
# )

# COMMAND ----------

# MAGIC %md
# MAGIC Note: <b>In you decide to drop table from UI follow the follwing steps.</b>.
# MAGIC
# MAGIC Follow the following steps:
# MAGIC - Go to [Feature Store](/#feature-store/feature-store)
# MAGIC - Select the feature tables and select `delete` after clicking on 3 vertical dots icon.
# MAGIC
# MAGIC Deleting the feature tables in this way requires you to manually delete the published online tables and the underlying Delta table separately. 

# COMMAND ----------


