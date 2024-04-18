import os
import csv
from datetime import datetime, timedelta

"""
    A class used to parse ICU data files, and produce a final dataset file.
"""


class ICUDataParser:
    def __init__(self, logger):
        self.logger = logger
        # final data list to store the final dataset
        self.final_data = []

        """
        patient data map to store the timestamp and values for each column of patient
        each of the map items will look like:
        
        {
            "458":{
               "values":{
                  "2012-10-09 06:00:00":{
                     "glucose":{
                        "value":"100.0",
                        "num_records_processed":3
                     },
                     "haemoglobin":{
                        "value":"12.8",
                        "num_records_processed":1
                     },
                  }
                  "2012-10-09 06:30:00":{
                     "glucose":{
                        "value":"110.0",
                        "num_records_processed":2
                     },
                     "haemoglobin":{
                        "value":"10.4",
                        "num_records_processed":3
                     },
                  }
               }
               "date_of_birth": "1989-01-01"
            }
        }
        """
        self.patient_data_map = {}
        # add the final data columns (headers) here
        headers = [
            "patient_id",
            "date_of_birth",
            "timestamp",
            "cpp",
            "glucose",
            "haemoglobin",
            "heart_rate",
            "icp",
            "mean_blood_pressure",
            "paco2",
            "pao2",
            "peep",
            "ph",
            "respiration_rate",
            "spo2",
            "temperature",
        ]
        self.final_data.append(headers)
        self.column_index = {header: index for index, header in enumerate(headers)}

    def parse(self, files, data_dir):
        self.logger.debug(f"Found {len(files)} files to parse")
        for file in files:
            file_name = os.path.basename(file)
            self.logger.debug(f"Parsing file {file_name} ...")
            if "episodes with high icp.txt" in file_name:
                self.parse_episodes_with_high_icp(file)
            elif file_name.endswith(".txt"):
                self.parse_file(file, file_name[:-4].lower().replace(" ", "_"))
        # Create the final data
        self.create_final_data()
        # save the final data to a file
        self.save_final_data(data_dir)

    def parse_file(self, file, column_name):
        with open(file, "r", encoding='utf-8-sig') as f:
            for line in Helpers.non_blank_lines(f):
                self.logger.debug(f"Processing line: {line}")
                patient_id, timestamp, _, value = line.split(",")
                patient_id = patient_id.strip().replace('"', "")
                timestamp = timestamp.strip().replace('"', "")

                # Convert the timestamp string to a datetime object
                timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")

                # Round the timestamp to the nearest half hour
                timestamp = Helpers.round_time(timestamp)

                value = value.strip().replace('"', "")
                self.add_data_value_for_patient_at_timestamp(
                    patient_id, timestamp, column_name, value
                )

    def parse_episodes_with_high_icp(self, file):
        with open(file, "r", encoding='utf-8-sig') as f:
            for line in Helpers.non_blank_lines(f):
                self.logger.debug(f"Processing line: {line}")
                patient_id, date_of_birth, _, _, _, _, _, _ = line.split(",")
                patient_id = patient_id.strip().replace('"', "")
                date_of_birth = date_of_birth.strip().replace('"', "")
                self.add_date_of_birth_for_patient(patient_id, date_of_birth)

    def add_data_value_for_patient_at_timestamp(
            self, patient_id, timestamp, data_type, value
    ):
        # check if the patient exists in the map
        if patient_id not in self.patient_data_map:
            self.patient_data_map[patient_id] = {
                "values": {},
            }

        # check if the timestamp exists for the patient in the map
        if timestamp not in self.patient_data_map[patient_id]["values"]:
            self.patient_data_map[patient_id]["values"][timestamp] = {}

        # check if the data type exists for the timestamp of the patient in the map
        if data_type not in self.patient_data_map[patient_id]["values"][timestamp]:
            self.patient_data_map[patient_id]["values"][timestamp][data_type] = {
                'value': value,
                'num_records_processed': 1
            }
        else:
            # if the data type already exists, update the value with the average and increment the count
            current_value = self.patient_data_map[patient_id]["values"][timestamp][data_type]['value']
            num_of_records_processed = self.patient_data_map[patient_id]["values"][timestamp][data_type][
                                           'num_records_processed'] + 1
            average_value = float(current_value) + (float(value) - float(current_value)) / num_of_records_processed
            self.patient_data_map[patient_id]["values"][timestamp][data_type]['value'] = average_value
            self.patient_data_map[patient_id]["values"][timestamp][data_type][
                'num_records_processed'] = num_of_records_processed

    def add_date_of_birth_for_patient(self, patient_id, date_of_birth):
        # check if the patient exists in the map
        if patient_id not in self.patient_data_map:
            self.patient_data_map[patient_id] = {}

        # add the date of birth to the patient data in the map
        self.patient_data_map[patient_id]["date_of_birth"] = date_of_birth

    def create_final_data(self):
        # Iterate over the patient_data_map
        for patient_id, patient_data in self.patient_data_map.items():
            for timestamp, data_types in patient_data["values"].items():
                # Create a new row for each timestamp, initialize it with -1
                row = [-1] * len(self.final_data[0])
                row[self.column_index['patient_id']] = patient_id
                row[self.column_index['timestamp']] = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                # Add the data values for each data type
                for data_type, data in data_types.items():
                    row[self.column_index[data_type]] = data['value']
                # Add the date of birth
                row[self.column_index['date_of_birth']] = patient_data['date_of_birth']
                # Add the row to the final_data list
                self.final_data.append(row)

    def save_final_data(self, data_dir, file_name="final_data.csv"):
        file = data_dir + os.sep + file_name
        with open(file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.final_data)


class Helpers:
    @staticmethod
    def round_time(dt, minutes=30):
        """
        Rounds a datetime object to its nearest half hour.
        """
        round_to = timedelta(minutes=minutes)
        seconds = (dt.replace(tzinfo=None) - dt.min).seconds
        rounding = (seconds + round_to.seconds / 2) // round_to.seconds * round_to.seconds
        return dt + timedelta(0, rounding - seconds, -dt.microsecond)

    @staticmethod
    def non_blank_lines(f):
        for line in f:
            line = line.rstrip()
            if line:
                yield line
