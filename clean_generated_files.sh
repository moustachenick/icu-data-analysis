#!/bin/bash
# Remove all generated files

echo "Removing all generated files..."

rm data/cleaned_df_lagged.csv
rm data/final_data.csv
rm data/test_data.csv
rm data/test_data_classification.csv
rm data/train_data.csv
rm data/train_data_classification.csv

echo "All generated files removed ✅"