import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer


# Function to print the percentage of rows that have more than 1 column with missing values,
# and the percentage of rows that have exactly 1 column with missing values.
def print_percentages_of_rows_with_missing_values(dataframe):
    # count the number of rows that have more than 1 column with a missing value
    number_of_rows_with_more_than_1_missing_value = dataframe[dataframe.isna().sum(axis=1) > 1].shape[0]
    # print the percentage of rows that have more than 1 column with missing values
    print('Percentage of rows that have more than 1 column with missing values: ',
          number_of_rows_with_more_than_1_missing_value / total_num_of_rows * 100)
    # number of rows with exactly 1 missing value
    number_of_rows_with_1_missing_value = dataframe[dataframe.isna().sum(axis=1) == 1].shape[0]
    # print the percentage of rows that have exactly 1 column with value -1
    print('Percentage of rows that have exactly 1 column with missing values: ',
          number_of_rows_with_1_missing_value / total_num_of_rows * 100)


# read the data from the csv file
df = pd.read_csv('../../../data/final_data.csv', engine='python')
df.head()

# The dataset contains missing values that are represented by different strings ("--", ".", etc).
# declare an array of strings that will be converted to NaN
missing_values_representations = [-1, '-1', '--', '-', '.', "/"]

# Replace missing values with NaN
df.replace(missing_values_representations, np.nan, inplace=True)

# print the total number of rows
total_num_of_rows = df.shape[0]
print()
print('Total number of rows: ', total_num_of_rows)
print()
print('Percentage of rows with missing values initially:')

print_percentages_of_rows_with_missing_values(df)

# We want to fill the missing values in the data.
# We can find the nearest neighbor to the row with the missing value and replace it with that value.

# drop the 'respiration_rate' column as it has many missing values, and is not needed for the analysis
df.drop(columns=['respiration_rate'], inplace=True)
print()

print('DataFrame after dropping the "respiration_rate" column:')

print_percentages_of_rows_with_missing_values(df)

# Let's drop the rows with more than 1 missing value
missing_count_per_row = df.isnull().sum(axis=1)

filtered_df = df[missing_count_per_row <= 1]

print()
print('DataFrame after dropping rows with more than 1 missing value:')

print_percentages_of_rows_with_missing_values(filtered_df)

# A percentage of the rows have exactly 1 missing value.
# We will fill these missing values, by finding the nearest neighbor.

# Separate the "date" columns from the rest of the data (we will add them back later)
date_cols = ['date_of_birth', 'timestamp']

date_data = filtered_df[date_cols]
numeric_df = filtered_df.drop(columns=date_cols)

missing_count_per_row = numeric_df.isnull().sum(axis=1)

# Split the DataFrame into 2 parts: one with no missing values and one with exactly one missing value
# Rows with no missing values
no_missing_df = numeric_df[missing_count_per_row == 0]

# Rows with exactly one missing value
one_missing_df = numeric_df[missing_count_per_row == 1]

# Initialize KNNImputer with a small number of neighbors (e.g., 1 or 2)
# This will find the nearest neighbor to the row with the missing value and replace it with that value.
imputer = KNNImputer(n_neighbors=1)

# Impute missing values in the DataFrame with one missing value
imputed_one_missing = imputer.fit_transform(one_missing_df)

# Convert the imputed array back to a DataFrame
imputed_one_missing_df = pd.DataFrame(imputed_one_missing, columns=numeric_df.columns, index=one_missing_df.index)

# Combine the DataFrames back together
combined_df = pd.concat([no_missing_df, imputed_one_missing_df])

# Sort the combined DataFrame to maintain the original order
combined_df = combined_df.sort_index()

# Add the date columns back to the combined DataFrame
combined_df = pd.concat([combined_df, date_data], axis=1)

print()
print('DataFrame after imputing the missing values:')
print_percentages_of_rows_with_missing_values(combined_df)


