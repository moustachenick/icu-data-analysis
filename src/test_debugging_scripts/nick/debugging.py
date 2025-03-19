import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pandas as pd
from data_parser.data_pre_processor import DataPreProcessor

file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'final_data.csv'))

data_processor = DataPreProcessor(file_path)

filtered_df = data_processor.standardize_missing_values()

combined_df = data_processor.known_nearest_neighbor_imputer(filtered_df)

# Save the combined DataFrame to a CSV file
combined_df.to_csv('combined.csv', index=False)

# Ensure 'icp' is a numeric column
combined_df['icp'] = pd.to_numeric(combined_df['icp'], errors='coerce')

# Count the number of negative 'icp' values
negative_icp_count = (combined_df['icp'] < 0).sum()

print(f"Number of negative ICP values: {negative_icp_count}")
print("Total Rows =", len(combined_df))