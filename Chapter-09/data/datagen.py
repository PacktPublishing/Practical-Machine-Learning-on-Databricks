# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ### Month 1 - Base line Data
# MAGIC
# MAGIC We will generate a dummy dataset for showcasing model drift. The dataset consists of time series data for 3 months. 
# MAGIC
# MAGIC The independent features of the dataset include the following features:
# MAGIC
# MAGIC **Features**
# MAGIC * `Temperature` (Numeric) : Highest daily temperature in Fahrenheit. 
# MAGIC * `Weather_Condition` (Categorical): 'sunny', 'cloudy', 'rainy' 
# MAGIC * `Promotion_Type` (Categorical): 'discount', 'free_gift', 'bundle_deal'
# MAGIC * `Website_Traffic` (Numeric): Total website traffic
# MAGIC * `Device_Type` (Categorical): 
# MAGIC
# MAGIC **Target**
# MAGIC * `Daily_Sales` (Numeric):  
# MAGIC
# MAGIC The `Daily_Sales` target will have following correlation with various features"
# MAGIC * `Positive correlation` with `Temperature` and `Website_Traffic`.
# MAGIC * `Negative correlation` with `Weather_Condition` and `Device_Type`.
# MAGIC
# MAGIC We will train our model on the first month worth of data and then simulate various drift patterns in the consecutive months of data. 
# MAGIC

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(0)

# Generate dates for the time series data
dates = pd.date_range('2023-01-01', '2023-01-31')
num_days = len(dates)
# Generate independent feature data
temperature = np.round(np.random.normal(loc=25, scale=5, size=num_days), 2)
weather_condition = np.random.choice(['sunny', 'cloudy', 'rainy'], size=num_days, p=[0.5, 0.3, 0.2])
promotion_type = np.random.choice(['discount', 'free_gift', 'bundle_deal'], size=num_days, p=[0.4, 0.3, 0.3])
website_traffic = np.random.normal(loc=500, scale=100, size=num_days).astype(int)  # Generate website traffic as integers
device_type = np.random.choice(['mobile', 'desktop', 'tablet'], size=num_days, p=[0.6, 0.3, 0.1])

# Generate dependent feature data (daily sales)
# Add positive correlation with temperature and website_traffic
# Add negative correlation with weather_condition and device_type
sales = np.round(1000 + 10*temperature + 5*website_traffic - 50*(weather_condition == 'rainy') - 100*(device_type == 'desktop')).astype(int)

# Create a pandas DataFrame to store the time series data
sales_data_month1 = pd.DataFrame({'Date': dates,
                          'Temperature': temperature,
                          'Weather_Condition': weather_condition,
                          'Promotion_Type': promotion_type,
                          'Website_Traffic': website_traffic,
                          'Device_Type': device_type,
                          'Daily_Sales': sales})


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Month 2 - New Data Arrives
# MAGIC
# MAGIC Our model has been deployed for a month and we now have an incoming fresh month of data.
# MAGIC
# MAGIC **Scenario:**
# MAGIC * An updated upstream Data cleaning process has a bug causing the the value of, `website_traffic` counts for promotion type `bundle_deal` and `free_gift` to be empty.  
# MAGIC
# MAGIC * Also during the upstream data generation procedure a the temperature values are now being captured in __Fahrenheit__ rather than in __Celcius__.
# MAGIC   
# MAGIC **What are we simulating here?**
# MAGIC * Feature drift
# MAGIC * Upstream data errors

# COMMAND ----------

# Generate dates for the time series data
dates = pd.date_range('2023-02-01', '2023-02-28')
num_days = len(dates)

# introducing feature drift
# Generate independent feature data
temperature_celcicus = np.round(np.random.normal(loc=25, scale=5, size=num_days), 2)

weather_condition = np.random.choice(['sunny', 'cloudy', 'rainy'], size=num_days, p=[0.5, 0.3, 0.2])
promotion_type = np.random.choice(['discount', 'free_gift', 'bundle_deal'], size=num_days, p=[0.4, 0.3, 0.3])
website_traffic = np.random.normal(loc=500, scale=100, size=num_days).astype(int)  # Generate website traffic as integers
device_type = np.random.choice(['mobile', 'desktop', 'tablet'], size=num_days, p=[0.6, 0.3, 0.1])

# Generate dependent feature data (daily sales)
# Add positive correlation with temperature and website_traffic
# Add negative correlation with weather_condition and device_type
sales = np.round(1000 + 10*temperature_celcicus + 5*website_traffic - 50*(weather_condition == 'rainy') - 100*(device_type == 'desktop')).astype(int)

# Create a pandas DataFrame to store the time series data
sales_data_month2_correct = pd.DataFrame({'Date': dates,
                          'Temperature': temperature_celcicus,
                          'Weather_Condition': weather_condition,
                          'Promotion_Type': promotion_type,
                          'Website_Traffic': website_traffic,
                          'Device_Type': device_type,
                          'Daily_Sales': sales})


#change temperature scale to Fehrenheit
#Convert the Celsius temperatures to Fahrenheit
temperature_fahrenheit = (temperature_celcicus * 9 / 5) + 32


# Create a pandas DataFrame to store the time series data
sales_data_month2_wrong = pd.DataFrame({'Date': dates,
                          'Temperature': temperature_fahrenheit,
                          'Weather_Condition': weather_condition,
                          'Promotion_Type': promotion_type,
                          'Website_Traffic': website_traffic,
                          'Device_Type': device_type,
                          'Daily_Sales': sales})

#introducing upstream processing error causing website traffic to be empty for bundle_deal and free_gift
sales_data_month2_wrong.loc[sales_data_month2_wrong['Promotion_Type'] == 'bundle_deal', 'Website_Traffic'] = None
sales_data_month2_wrong.loc[ sales_data_month2_wrong['Promotion_Type'] == 'free_gift', 'Website_Traffic'] = None

sales_data_month2_wrong.to_csv(f'/dbfs{raw_month2_bad_data_path}/data.csv', index=False)

# COMMAND ----------

#sales_data_month2_correct

# COMMAND ----------

# MAGIC %md
# MAGIC ### Month 3
# MAGIC
# MAGIC **Scenario:**
# MAGIC * A product campaign went viral on social media. Sales increased by 30% for each day. 
# MAGIC   
# MAGIC **What are we simulating here?**
# MAGIC * Concept Drift

# COMMAND ----------

dates = pd.date_range('2023-03-01', '2023-03-31')
num_days = len(dates)

# Generate independent feature data
temperature = np.round(np.random.normal(loc=25, scale=5, size=num_days), 2)
weather_condition = np.random.choice(['sunny', 'cloudy', 'rainy'], size=num_days, p=[0.5, 0.3, 0.2])
promotion_type = np.random.choice(['discount', 'free_gift', 'bundle_deal'], size=num_days, p=[0.4, 0.3, 0.3])
website_traffic = np.random.normal(loc=500, scale=100, size=num_days).astype(int)  # Generate website traffic as integers
device_type = np.random.choice(['mobile', 'desktop', 'tablet'], size=num_days, p=[0.6, 0.3, 0.1])

#increase daily sales by 30%
sales = np.round((1000 - 10*temperature + 5*website_traffic - 50*(weather_condition == 'rainy') - 100*(device_type == 'desktop')) * 1.3).astype(int)

# Create a pandas DataFrame to store the time series data
sales_data_month3 = pd.DataFrame({'Date': dates,
                          'Temperature': temperature,
                          'Weather_Condition': weather_condition,
                          'Promotion_Type': promotion_type,
                          'Website_Traffic': website_traffic,
                          'Device_Type': device_type,
                          'Daily_Sales': sales})


#sales_data_month3

# COMMAND ----------

merged_raw_df = pd.concat([sales_data_month1, sales_data_month2_correct, sales_data_month3])
# Write the dataframe to a CSV file and give path to dbfs directory we created for storing the raw file.
merged_raw_df.to_csv(f'/dbfs{raw_good_data_path}/data.csv', index=False)
