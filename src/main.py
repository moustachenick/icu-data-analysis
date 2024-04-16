import os
import argparse
import logging
from logging_custom_formatter import CustomFormatter
from icu_data_parser.icu_data_parser import ICUDataParser

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()
parser.add_argument('--directory', '-dir', help="the directory to scan for files", type=str, default="../data")
parser.add_argument('--log', '-l', help="log level (DEBUG, WARN, ERROR)", type=str, default="DEBUG")
parser.add_argument('--dir-relative', '-dr', help="directory path is relative", type=bool, default="True")


def main(data_dir):
    logger.debug(f"Scanning directory {data_dir}")
    files_to_parse = []
    icu_data_parser = ICUDataParser(logger)
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            logger.debug(f"Found file {root + os.sep + file}")
            if file.endswith(".txt"):
                files_to_parse.append(root + os.sep + file)
    icu_data_parser.parse(files_to_parse)
    return 0


def set_up_logger(log_level):
    # Create a handler
    c_handler = logging.StreamHandler()
    # Create a formatter and attach it to the handler
    c_handler.setFormatter(CustomFormatter())
    # link handler to logger
    logger.addHandler(c_handler)
    # Set logging level to the logger
    logger.setLevel(log_level)


if __name__ == '__main__':
    args = parser.parse_args()
    set_up_logger(args.log.upper())
    directory = args.directory
    if directory is None:
        raise Exception("root directory argument is required")
    if args.dir_relative:
        directory = os.path.join(os.path.abspath(os.path.dirname(__file__)), directory)
    main(directory)
