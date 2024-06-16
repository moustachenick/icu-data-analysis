import pandas as pd
import numpy as np


file_path = r'C:\Users\Nick\Desktop\giteroo\icu-data-analysis\data\final_data.csv'
df = pd.read_csv(file_path,low_memory=False)

# Ensure all `-1` strings are converted to integers
df.replace('-1', -1, inplace=True)

# Replace all negative values with `-1`
df = df.applymap(lambda x: -1 if isinstance(x, (int, float)) and x < 0 else x)

# Check for lines with -1 values and count how many lines have more than one -1 value
df['missing_count'] = (df == -1).sum(axis=1)
lines_with_more_than_one_missing_value = df[df['missing_count'] > 1].shape[0]
total_lines = df.shape[0]
percentage_lines_with_more_than_one_missing = (lines_with_more_than_one_missing_value / total_lines) * 100

# Display the results
print(f"Number of lines with more than one missing value: {lines_with_more_than_one_missing_value}")
print(f"Percentage of lines with more than one missing value: {percentage_lines_with_more_than_one_missing:.2f}%")

# Filter out line with more than one missing value
df_filtered = df[df['missing_count'] <= 1].copy()

# Drop the 'missing_count' column 
df_filtered.drop(columns=['missing_count'], inplace=True)


# Extract date from the timestamp for grouping
df_filtered['date'] = pd.to_datetime(df_filtered['timestamp']).dt.date

# Replace -1 with NaN for consistency in identifying missing values
df_filtered.replace(-1, np.nan, inplace=True)

# Display the first few rows before filling missing values
print("DataFrame before filling missing values:")
print(df_filtered.head())

# Define a function to interpolate within each group
def interpolate_group(group):
    return group.interpolate(method='nearest')

# Group by patient_id and date, and apply the interpolation function
df_filled = df_filtered.groupby(['patient_id', 'date']).apply(interpolate_group)

# Drop the 'date' column as it's no longer needed
df_filled.drop(columns=['date'], inplace=True)

# Replace NaN back with -1 if necessary
df_filled.replace(np.nan, -1, inplace=True)

# Optionally, export the filled DataFrame to a new CSV file
filled_file_path = r'C:\Users\Nick\Desktop\giteroo\icu-data-analysis\data\nikos_filled_data.csv'
df_filled.to_csv(filled_file_path, index=False)