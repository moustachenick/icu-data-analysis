
class ICUDataParser:
    def __init__(self, logger):
        self.logger = logger

    def parse(self, files):
        self.logger.debug(f"Found {len(files)} files to parse")