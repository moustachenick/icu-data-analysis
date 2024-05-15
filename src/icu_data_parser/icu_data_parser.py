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
        self.last_known_values = {}

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
                timestamp = Helpers.custom_round_time(timestamp)

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
    
    def get_last_known_value(self, patient_id, data_type):
        """
        Retrieves the last known value and its timestamp for a given patient and data type.
        Returns a tuple of (value, timestamp).
        """
        return self.last_known_values.get(patient_id, {}).get(data_type, ("", None))

    def update_last_known_value(self, patient_id, data_type, value, timestamp):
        """
        Updates the tracking of the last known value and its timestamp for a given patient and data type.
        """
        if patient_id not in self.last_known_values:
            self.last_known_values[patient_id] = {}
        self.last_known_values[patient_id][data_type] = (value, timestamp)

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
        
         # Forward fill missing values 
        if value == "-1":
            last_value, last_timestamp = self.get_last_known_value(patient_id, data_type)
            if last_timestamp and (timestamp - last_timestamp <= timedelta(minutes=30)):
             value = last_value
            else:
             value = "-1"

         # Update last known value only if it's not missing
        if value != "-1":
            self.update_last_known_value(patient_id, data_type, value, timestamp)

        # Check for special handling variables
        special_handling_variables = ["glucose", "haemoglobin", "paco2", "pao2", "ph"]
        if data_type in special_handling_variables and value == "-1":
            last_value, last_timestamp = self.get_last_known_value(patient_id, data_type)
            # Check if the last known value is from the same day
            if last_timestamp and last_timestamp.date() == timestamp.date():
                 value = last_value
            else:
                 value = "-1"  # No valid last known value from the same day


    def add_date_of_birth_for_patient(self, patient_id, date_of_birth):
        # check if the patient exists in the map
        if patient_id not in self.patient_data_map:
            self.patient_data_map[patient_id] = {}

        # add the date of birth to the patient data in the map
        self.patient_data_map[patient_id]["date_of_birth"] = date_of_birth

    def create_final_data(self):
        # Iterate over the patient_data_map
        for patient_id, patient_data in self.patient_data_map.items():
            sorted_timestamps = sorted(patient_data["values"].keys())
            for timestamp in sorted_timestamps:
                data_types = patient_data["values"][timestamp]
                # Create a new row for each timestamp, initialize it with -1
                row = [-1] * len(self.final_data[0])
                row[self.column_index['patient_id']] = patient_id
                row[self.column_index['timestamp']] = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                # Add the data values for each data type
                for data_type, data in data_types.items():
                    row[self.column_index[data_type]] = data['value']
                current_value = data_types.get(data_type, {'value': '-1'})['value']    
                row[self.column_index[data_type]] = current_value if current_value != '-1' else "-1"

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
    def custom_round_time(dt):
        """
        Custom rounding:
        - :46 to :15 -> round down to :00 of the current hour
        - :16 to :45 -> round up to :30 of the current hour
        """
        if dt.minute >= 46 or dt.minute <= 15:
            # Round down to the hour
            return dt.replace(minute=0, second=0, microsecond=0)
        elif 16 <= dt.minute <= 45:
            # Round to half past the hour
            return dt.replace(minute=30, second=0, microsecond=0)
        return dt

    @staticmethod
    def non_blank_lines(f):
        for line in f:
            line = line.rstrip()
            if line:
                yield line
