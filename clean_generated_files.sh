#!/bin/bash
# Remove all generated files

echo "Removing all generated files..."

rm -f data/cleaned_df_lagged*.csv
rm -f data/cleaned_df_*.csv
rm -f data/final_data.csv
rm -f data/test_data*.csv
rm -f data/train_data*.csv
rm -f data/validation_data*.csv

echo "All generated files removed ✅"