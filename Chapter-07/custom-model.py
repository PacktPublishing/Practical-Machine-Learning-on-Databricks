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
# MAGIC
# MAGIC ## Predicting Wine Cultivars using Decision Tree Classifier and MLflow
# MAGIC
# MAGIC This code is designed to solve a multi-class classification problem using the wine dataset. The wine dataset contains 178 samples, each belonging to one of three different cultivars (types of grape) in Italy. Each sample has 13 different features like Alcohol, Malic acid, etc.
# MAGIC
# MAGIC ### Objective
# MAGIC
# MAGIC The objective of the model is to predict the cultivar to which a given wine sample belongs based on its 13 features. In simpler terms, for a new wine sample, the model aims to categorize it as 'class_0', 'class_1', or 'class_2', representing one of the three possible cultivars. Additionally, the model provides the probabilities for the sample belonging to each of these classes.
# MAGIC
# MAGIC ### Implementation
# MAGIC
# MAGIC The code uses a Decision Tree classifier and trains it on a subset of the wine dataset, known as the training set. After training, the model is encapsulated in a custom Python class (`CustomModelWrapper`). This class facilitates the logging of the model using MLflow, a platform for end-to-end machine learning lifecycle management.
# MAGIC
# MAGIC Once the model is logged, it can be deployed and used to make predictions on new, unseen data, commonly referred to as the test set.

# COMMAND ----------

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
import mlflow
import mlflow.pyfunc
import pandas as pd

# Custom model class
class CustomModelWrapper(mlflow.pyfunc.PythonModel):
    # Initialize the classifier model in the constructor
    def __init__(self, classifier_model):
        self.classifier_model = classifier_model

    # Prediction method
    def predict(self, context, model_data):
        # Compute the probabilities and the classes
        probs = self.classifier_model.predict_proba(model_data)
        preds = self.classifier_model.predict(model_data)
        
        # Create a DataFrame to hold probabilities and predictions
        labels = ["class_0", "class_1", "class_2"]
        result_df = pd.DataFrame(probs, columns=[f'prob_{label}' for label in labels])
        result_df['prediction'] = [labels[i] for i in preds]
        
        return result_df

# Load the wine dataset and split it into training and test sets
wine_data = load_wine()
X, y = wine_data.data, wine_data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)

# Initialize and fit the DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(random_state=7)
dt_classifier.fit(X_train, y_train)

# Create an instance of the CustomModelWrapper
custom_wrapper = CustomModelWrapper(dt_classifier)

# Define the input and output schema
input_cols = [ColSpec("double", feature) for feature in wine_data.feature_names]
output_cols = [ColSpec("double", f'prob_{cls}') for cls in wine_data.target_names] + [ColSpec("string", 'prediction')]
model_sign = ModelSignature(inputs=Schema(input_cols), outputs=Schema(output_cols))

# Prepare an example input
input_sample = pd.DataFrame(X_train[:1], columns=wine_data.feature_names)
input_sample_dict = input_sample.to_dict(orient='list')

# Log the model using MLflow
with mlflow.start_run():
    mlflow.pyfunc.log_model("wine_model",
                            python_model=custom_wrapper,
                            input_example=input_sample_dict,
                            signature=model_sign)

# Retrieve the run ID and load the logged model
last_run_id = mlflow.last_active_run().info.run_id
retrieved_model = mlflow.pyfunc.load_model(f"runs:/{last_run_id}/wine_model")

# Create a DataFrame for the test data
test_df = pd.DataFrame(X_test[:1], columns=wine_data.feature_names)

# Use the loaded model for prediction
prediction_result = retrieved_model.predict(test_df)


# COMMAND ----------

prediction_result

# COMMAND ----------


