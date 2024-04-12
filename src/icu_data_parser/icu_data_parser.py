import os
import csv

"""
    A class used to parse ICU data files, and produce a final dataset file.
"""


def non_blank_lines(f):
    for line in f:
        line = line.rstrip()
        if line:
            yield line


class ICUDataParser:
    def __init__(self, logger):
        self.logger = logger
        # final data list to store the final dataset
        self.final_data = []
        # add the final data columns (headers) here
        headers = ["patient_id", "timestamp", "cpp", "glucose", "haemoglobin", "heart_rate", "icp",
                   "mean_blood_pressure", "paco2", "pao2", "peep", "ph", "respiration_rate", "spo2",
                   "temperature"]
        self.final_data.append(headers)
        self.column_index = {header: index for index, header in enumerate(headers)}

    def parse_files(self, files):
        self.logger.debug(f"Found {len(files)} files to parse")
        for file in files:
            file_name = os.path.basename(file)
            self.logger.debug(f"Parsing file {file_name} ...")
            if file_name.endswith(".txt"):
                self.parse_file(file, file_name[:-4].lower().replace(" ", "_"))
        # save the final data to a file
        self.save_final_data()

    def parse_file(self, file, column_name):
        with open(file, 'r') as f:
            for line in non_blank_lines(f):
                print(line)
                patient_id, timestamp, _, value = line.split(",")
                patient_id = patient_id.strip().replace('"', "")
                timestamp = timestamp.strip().replace('"', "")
                value = value.strip().replace('"', "")
                self.add_data_value_for_patient_at_timestamp(patient_id, timestamp, column_name, value)

    def add_data_value_for_patient_at_timestamp(self, patient_id, timestamp, data_type, value):
        # find the patient in the final data list
        for data in self.final_data:
            if data[:2] == [patient_id, timestamp]:
                data[self.column_index[data_type]] = value
                return

        # if the patient/timestamp record does not exist, we need to add the new record

        # Create a new list with all elements being -1
        new_element = [-1] * len(self.final_data[0])
        # Set patient_id and the timestamp
        new_element[self.column_index["patient_id"]] = patient_id
        new_element[self.column_index["timestamp"]] = timestamp

        # Set the value of the desired column
        new_element[self.column_index[data_type]] = value
        # Append the new element to final_data
        self.final_data.append(new_element)

    def save_final_data(self, filename='final_data.csv'):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self.final_data)
