import sys
import os
import multiprocessing

from matplotlib import pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pandas as pd
import numpy as np
from icu_data_parser.DataPreProcessor import DataPreProcessor





data_processor = DataPreProcessor()

filtered_df = data_processor.replace_missing_values()

combined_df = data_processor.known_nearest_neighbor_imputer(filtered_df)

# Save the combined DataFrame to a CSV file
combined_df.to_csv('combined.csv', index=False)


# Ensure 'icp' is a numeric column
combined_df['icp'] = pd.to_numeric(combined_df['icp'], errors='coerce')

# Count the number of negative 'icp' values
negative_icp_count = (combined_df['icp'] < 0).sum()

print(f"Number of negative ICP values: {negative_icp_count}")
print("Total Rows =", len(combined_df))