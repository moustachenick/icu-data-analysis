# main.py
import os

from icu_data_parser.DataParser import DataParser
from icu_icp_predictions.ICPPredictor import ICPPredictor


def main():
    """
    Main function to initialize and run the ICPPrediction pipeline.
    """
    # Path for the raw data directory (icu-data-analysis/data)
    raw_data_dir_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data")
    )

    # Use the DataParser class to process the raw data
    data_parser = DataParser(raw_data_dir_path)
    raw_data_file_path = data_parser.run()

    # Initialize the ICPPrediction class
    predictor = ICPPredictor(raw_data_file_path)

    # Run the pipeline
    print("Running the ICP Prediction pipeline...\n")
    predictor.run()

if __name__ == "__main__":
    main()
