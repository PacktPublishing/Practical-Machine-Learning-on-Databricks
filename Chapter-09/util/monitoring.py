# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC #### Monitoring Utility Functions
# MAGIC
# MAGIC The following functions check
# MAGIC - the proportion of nulls
# MAGIC - the differences in summary statistics
# MAGIC - the shifts in distributions

# COMMAND ----------


from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

def check_null_proportion(new_pdf, null_proportion_threshold):
  """
  Function to compute the proportions of nulls for all columns in a Spark DataFrame and return any features that exceed the specified null threshold.

  Args:
    df: (pd.DataFrame) The DataFrame that contains new incoming data.
    null_proportion_threshold: (float) A numeric value ranging from 0 and 1 that specifies the tolerable fraction of nulls.

  Returns:
    A dictionary mapping feature names to their null proportions.

  Raises:
    ValueError: If the null proportion threshold is not between 0 and 1.

    Notes:
      * This function uses the `isnull()` method to identify null values in the DataFrame.
      * The `sum()` method is used to count the number of null values in each column.
      * The `len()` method is used to get the total number of rows in the DataFrame.
      * The `transpose()` method is used to convert the DataFrame from a long format to a wide format.
      * The `assert` statement is used to check that the null proportion threshold is between 0 and 1.
      * The `print()` statement is used to print an alert if there are any features that exceed the null proportion threshold.
  """

  # Check that the null proportion threshold is between 0 and 1.
  if null_proportion_threshold < 0 or null_proportion_threshold > 1:
    raise ValueError(
        "The null proportion threshold must be between 0 and 1. "
        f"Received: {null_proportion_threshold}"
    )

  # Compute the proportions of nulls for all columns in the DataFrame.
  missing_stats = pd.DataFrame(new_pdf.isnull().sum() / len(new_pdf)).transpose()

  # Get a list of the column names that exceed the null proportion threshold.
  null_col_list = missing_stats.columns[(missing_stats >= null_proportion_threshold).iloc[0]]

  # Create a dictionary mapping feature names to their null proportions.
  null_dict = {}
  for feature in null_col_list:
    null_dict[feature] = missing_stats[feature][0]

  # Check if any features exceed the null proportion threshold.
  if len(null_dict) > 0:
    print("Alert: There are feature(s) that exceed(s) the expected null threshold. Please ensure that the data is ingested correctly")
    print(null_dict)

  # Return the dictionary of null proportions.
  return null_dict


# COMMAND ----------

def check_diff_in_summary_stats(new_stats_pdf, prod_stats_pdf, num_cols, stats_threshold_limit, statistic_list):
  """
  Function to check if the new summary stats significantly deviates from the summary stats in the production data by a certain threshold.

  Args:
    new_stats_pdf: (pd.DataFrame) summary statistics of incoming data
    prod_stats_pdf: (pd.DataFrame) summary statistics of production data
    num_cols: (list) a list of numeric columns
    stats_threshold_limit: (float) a float < 1 that signifies the threshold limit
    statistic_list: (list) a list of statistics, e.g., mean, std, min, max

  Returns:
    A list of feature names that significantly deviate from the production data.

  Raises:
    ValueError: If the stats_threshold_limit is not between 0 and 1.

  Notes:
    * This function uses the `loc` method to get the value of a specific statistic for a given feature.
    * The `round` method is used to round a number to a specified number of decimal places.
    * The `print` statement is used to print the results of the function.
  """

  # Check that the stats_threshold_limit is between 0 and 1.
  if stats_threshold_limit < 0 or stats_threshold_limit > 1:
    raise ValueError(
        "The stats_threshold_limit must be between 0 and 1. "
        f"Received: {stats_threshold_limit}"
    )

  # Create a list of feature names that significantly deviate from the production data.
  feature_diff_list = []

  # Iterate over the numeric columns.
  for feature in num_cols:

    # Print a message indicating that the feature is being checked.
    print(f"\nCHECKING {feature}.........")

    # Iterate over the statistics.
    for statistic in statistic_list:

      # Get the value of the statistic for the feature in the production data.
      prod_stat_value = prod_stats_pdf[[str(feature)]].loc[str(statistic)][0]

      # Calculate the upper and lower threshold limits for the statistic.
      upper_val_limit = prod_stat_value * (1 + stats_threshold_limit)
      lower_val_limit = prod_stat_value * (1 - stats_threshold_limit)

      # Get the value of the statistic for the feature in the new data.
      new_stat_value = new_stats_pdf[[str(feature)]].loc[str(statistic)][0]

      # Check if the new statistic value is outside of the threshold limits.
      if new_stat_value < lower_val_limit:
        feature_diff_list.append(str(feature))
        print(f"\tThe {statistic} {feature} in the new data is at least {stats_threshold_limit * 100}% lower than the {statistic} in the production data. Decreased from {round(prod_stat_value, 2)} to {round(new_stat_value, 2)}.")

      elif new_stat_value > upper_val_limit:
        feature_diff_list.append(str(feature))
        print(f"\tThe {statistic} {feature} in the new data is at least {stats_threshold_limit * 100}% higher than the {statistic} in the production data. Increased from {round(prod_stat_value, 2)} to {round(new_stat_value, 2)}.")

  # Return the list of feature names that significantly deviate from the production data.
  return np.unique(feature_diff_list)


# COMMAND ----------

def check_diff_in_variances(reference_df, new_df, num_cols, p_threshold):
  """
  Function to check if the variances of the numeric columns in `new_df` are significantly different from the variances of the corresponding columns in `reference_df`.

  Args:
    reference_df: (pd.DataFrame) The DataFrame that contains the production data.
    new_df: (pd.DataFrame) The DataFrame that contains the new data.
    num_cols: (list) A list of the names of the numeric columns.
    p_threshold: (float) The p-value threshold for significance.

  Returns:
    A dictionary mapping feature names to their p-values.

  Raises:
    ValueError: If `p_threshold` is not between 0 and 1.

  Notes:
    * This function uses the `levene()` function from the `scipy.stats` module to perform the Levene test.
    * The `assert` statement is used to check that `p_threshold` is between 0 and 1.
    * The `print()` statements are used to print the results of the function.
  """

  # Check that `p_threshold` is between 0 and 1.
  if p_threshold < 0 or p_threshold > 1:
    raise ValueError(
        "The p_threshold must be between 0 and 1. "
        f"Received: {p_threshold}"
    )

  # Create a dictionary mapping feature names to their p-values.
  var_dict = {}

  # Iterate over the numeric columns.
  for feature in num_cols:

    # Perform the Levene test.
    levene_stat, levene_pval = stats.levene(reference_df[feature], new_df[feature], center="median")

    # If the p-value is less than or equal to the threshold, then the variances are significantly different.
    if levene_pval <= p_threshold:
      var_dict[feature] = levene_pval

  # Check if any features have significantly different variances.
  if len(var_dict) > 0:
    print(f"The feature(s) below have significantly different variances compared to production data at p-value {p_threshold}")
    print(var_dict)
  else:
    print(f"No features have significantly different variances compared to production data at p-value {p_threshold}")

  # Return the dictionary of p-values.
  return var_dict


# COMMAND ----------

def check_dist_ks_bonferroni_test(reference_df, new_df, num_cols, p_threshold, ks_alternative="two-sided"):
  """
  Function to take two pandas DataFrames and compute the Kolmogorov-Smirnov statistic on 2 sample distributions
  where the variable in question is continuous.
  This is a two-sided test for the null hypothesis that 2 independent samples are drawn from the same continuous
  distribution. If the KS statistic is small or the p-value is high, then we cannot reject the hypothesis that 
  the distributions of the two samples are the same.
  The alternative hypothesis can be either ‘two-sided’ (default), ‘less’ or ‘greater’.
  This function assumes that the distributions to compare have the same column name in both DataFrames.

  Args:
    reference_df: pandas DataFrame containing column with the distribution to be compared
    new_df: pandas DataFrame containing column with the distribution to be compared
    num_cols: (list) A list of the names of the numeric columns.
    p_threshold: (float) The p-value threshold for significance.
    ks_alternative: Defines the alternative hypothesis - ‘two-sided’ (default), ‘less’ or ‘greater’.

  Returns:
    A dictionary mapping feature names to their p-values.

  Raises:
    ValueError: If `p_threshold` is not between 0 and 1.

  Notes:
    * This function uses the `ks_2samp()` function from the `scipy.stats` module to perform the Kolmogorov-Smirnov test.
    * The `assert` statement is used to check that `p_threshold` is between 0 and 1.
    * The `print()` statements are used to print the results of the function.
    * The Bonferroni correction is used to adjust the p-value threshold to account for multiple comparisons.
  """

  # Check that `p_threshold` is between 0 and 1.
  if p_threshold < 0 or p_threshold > 1:
    raise ValueError(
        "The p_threshold must be between 0 and 1. "
        f"Received: {p_threshold}"
    )

  # Compute the Bonferroni-corrected alpha level.
  corrected_alpha = p_threshold / len(num_cols)

  # Create a dictionary mapping feature names to their p-values.
  ks_dict = {}

  # Iterate over the numeric columns.
  for feature in num_cols:

    # Compute the Kolmogorov-Smirnov statistic and p-value.
    ks_stat, ks_pval = stats.ks_2samp(reference_df[feature], new_df[feature], alternative=ks_alternative, mode="asymp")

    # If the p-value is less than or equal to the corrected alpha level, then the distributions are significantly different.
    if ks_pval <= corrected_alpha:
      ks_dict[feature] = ks_pval

  # Check if any features have significantly different distributions.
  if len(ks_dict) > 0:
    print(f"The feature(s) below have significantly different distributions compared to production data at Bonferroni-corrected alpha level of {round(corrected_alpha, 4)}, according to the KS test")
    print("\t", ks_dict)
  else:
    print(f"No feature distributions has shifted according to the KS test at the Bonferroni-corrected alpha level of {round(corrected_alpha, 4)}. ")

  # Return the dictionary of p-values.
  return ks_dict

# COMMAND ----------

def check_categorical_diffs(reference_pdf, new_pdf, cat_cols, p_threshold):
  """
  This function checks if there are differences in expected counts for categorical variables between the incoming data and the data in production.

  Args:
    reference_pdf: (pandas DataFrame) new incoming data
    new_pdf: (pandas DataFrame) data in production
    cat_cols: (list) a list of categorical columns

  Returns:
    A dictionary mapping feature names to their p-values.

  Raises:
    ValueError: If `p_threshold` is not between 0 and 1.

  Notes:
    * This function uses the `chisquare()` function from the `scipy.stats` module to perform the chi-squared test.
    * The `assert` statement is used to check that `p_threshold` is between 0 and 1.
    * The `print()` statements are used to print the results of the function.
  """

  # Check that `p_threshold` is between 0 and 1.
  if p_threshold < 0 or p_threshold > 1:
    raise ValueError(
        "The p_threshold must be between 0 and 1. "
        f"Received: {p_threshold}"
    )

  # Create a dictionary mapping feature names to their p-values.
  chi_dict = {}

  # Iterate over the categorical columns.
  for feature in cat_cols:

    # Calculate the observed frequencies by creating a contingency table using pd.crosstab
    observed_freq = pd.crosstab(reference_pdf[feature], new_pdf[feature])

    # Perform the Chi-Square test of independence
    chi2, p_value, _, _ = stats.chi2_contingency(observed_freq)

    # If the p-value is less than or equal to the threshold, then the expected counts are significantly different.
    if p_value <= p_threshold:
      chi_dict[feature] = p_value

  # Check if any features have significantly different expected counts.
  if len(chi_dict) > 0:
    print(f"The following categorical variables have significantly different expected counts compared to the production data at p-value {p_threshold}:")
    print("\t", chi_dict)
  else:
    print(f"No categorical variables have significantly different expected counts compared to the production data at p-value {p_threshold}.")

  return chi_dict

# COMMAND ----------

def compare_model_perfs(current_staging_run, current_prod_run, min_model_perf_threshold, metric_to_check):
  """
  This function compares the performances of the models in staging and in production.

  Args:
    current_staging_run: MLflow run that contains information on the staging model
    current_prod_run: MLflow run that contains information on the production model
    min_model_perf_threshold (float): The minimum threshold that the staging model should exceed before being transitioned to production
    metric_to_check (string): The metric that the user is interested in using to compare model performances

  Returns:
    Recommendation to transition the staging model to production or not

  Raises:
    ValueError: If `min_model_perf_threshold` is not positive.

  Notes:
    * This function uses the `data.metrics` attribute of the MLflow runs to get the metrics for the staging and production models.
    * The `round()` function is used to round the difference in performance to two decimal places.
    * The `print()` statements are used to print the results of the function.
  """

  # Check that `min_model_perf_threshold` is positive.
  if min_model_perf_threshold < 0:
    raise ValueError(
        "The min_model_perf_threshold must be positive. "
        f"Received: {min_model_perf_threshold}"
    )

  # Calculate the difference in performance between the staging and production models.
  model_diff_fraction = current_staging_run.data.metrics[str(metric_to_check)] / current_prod_run.data.metrics[str(metric_to_check)]
  model_diff_percent = round((model_diff_fraction - 1)*100, 2)

  # Print the performance of the staging and production models.
  print(f"Staging run's {metric_to_check}: {round(current_staging_run.data.metrics[str(metric_to_check)],3)}")
  print(f"Current production run's {metric_to_check}: {round(current_prod_run.data.metrics[str(metric_to_check)],3)}")

  # Recommend whether to transition the staging model to production.
  if model_diff_percent >= 0 and (model_diff_fraction - 1 >= min_model_perf_threshold):
    print(f"The current staging run exceeds the model improvement threshold of at least +{min_model_perf_threshold}. You may proceed with transitioning the staging model to production now.")
    
  elif model_diff_percent >= 0 and (model_diff_fraction - 1  < min_model_perf_threshold):
    print(f"CAUTION: The current staging run does not meet the improvement threshold of at least +{min_model_perf_threshold}. Transition the staging model to production with caution.")
  else: 
    print(f"ALERT: The current staging run underperforms by {model_diff_percent}% when compared to the production model. Do not transition the staging model to production.")


# COMMAND ----------

def plot_boxplots(unique_feature_diff_array, reference_pdf, new_pdf):
    """
    Plot boxplots comparing the distributions of unique features between incoming data and production data.

    Args:
        unique_feature_diff_array (list): List of unique feature names to compare.
        reference_pdf (pandas.DataFrame): Reference production data.
        new_pdf (pandas.DataFrame): New incoming data.

    Returns:
        matplotlib.figure.Figure: The generated figure.

    Raises:
        None

    """
    # Set the theme of the plots.
    sns.set_theme(style="whitegrid")

    # Calculate the number of columns.
    num_columns = len(unique_feature_diff_array)

    # Create a figure and axes.
    fig, axes = plt.subplots(1, num_columns, figsize=(5*num_columns, 5))

    # Set the title of the figure.
    fig.suptitle("Distribution Comparisons between Incoming Data and Production Data")

    # Plot boxplots for each column name side by side.
    for i, column_name in enumerate(unique_feature_diff_array):
        ax = axes[i] if num_columns > 1 else axes  # Access the correct subplot.
        ax.boxplot([reference_pdf[column_name], new_pdf[column_name]])
        ax.set_xticklabels(['Production Data', 'New Incoming Data'])
        ax.set_title(column_name)

    # Set common y-axis label.
    fig.text(0.04, 0.5, 'Value', va='center', rotation='vertical')

    # Set plot title.
    plt.suptitle('Boxplot Comparison')

    plt.close()

    # Return the generated figure.
    return fig

