import os


class ICUDataParser:
    def __init__(self, logger):
        self.logger = logger

    def parse(self, files):
        self.logger.debug(f"Found {len(files)} files to parse")
        for file in files:
            file_name = os.path.basename(file)
            self.logger.debug(f"Parsing file {file_name} ...")
            if file_name == "CPP.txt":
                self.parse_cpp_file(file)
            elif file_name == "episodes with high icp.txt":
                self.parse_ep_high_icp_file(file)
            elif file_name == "Glucose.txt":
                self.parse_glucose_file(file)
            elif file_name == "Hemoglobin.txt":
                self.parse_hemoglobin_file(file)
            elif file_name == "Hear rate.txt":
                self.parse_heart_rate_file(file)
            elif file_name == "ICP.txt":
                self.parse_icp_file(file)
            elif file_name == "Mean blood pressure.txt":
                self.parse_mean_blood_pressure_file(file)
            elif file_name == "PaCO2.txt":
                self.parse_paco2_file(file)
            elif file_name == "PaO2.txt":
                self.parse_pao2_file(file)
            elif file_name == "PEEP.txt":
                self.parse_peep_file(file)
            elif file_name == "PH.txt":
                self.parse_ph_file(file)
            elif file_name == "Respiration rate.txt":
                self.parse_respiration_rate_file(file)
            elif file_name == "SpO2.txt":
                self.parse_spo2_file(file)
            elif file_name == "Temperature.txt":
                self.parse_temperature_file(file)

    def parse_cpp_file(self, file):
        pass

    def parse_ep_high_icp_file(self, file):
        pass

    def parse_glucose_file(self, file):
        pass

    def parse_hemoglobin_file(self, file):
        pass

    def parse_heart_rate_file(self, file):
        pass

    def parse_icp_file(self, file):
        pass

    def parse_mean_blood_pressure_file(self, file):
        pass

    def parse_paco2_file(self, file):
        pass

    def parse_pao2_file(self, file):
        pass

    def parse_peep_file(self, file):
        pass

    def parse_ph_file(self, file):
        pass

    def parse_respiration_rate_file(self, file):
        pass

    def parse_spo2_file(self, file):
        pass

    def parse_temperature_file(self, file):
        pass
