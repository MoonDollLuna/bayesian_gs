# BAYESIAN NETWORK - NN GS SPEEDUP #
# Developed by Luna Jimenez Fernandez
# Based on the work of Wenfeng Zhang et al.

# IMPORTS #
import os.path
from pathlib import Path
import csv

import time

from typing import IO


class ResultsLogger:
    """
    "ResultsLogger" provides a wrapper for a CSV writer, containing all necessary methods to create and handle
    a CSV results file - including headers, column names, data writing and commentaries for final results.

    The resulting results file will have "<input_name>_<creation time>.csv" as a name.

    Parameters
    ----------
    results_path: str
        Location of the final CSV results file. Note that this path should NOT include the file name.
    input_name: str
        Name of the input data to use. Usually, the name of either the data file or the given dataset.
    flush_frequency: int
        How often the results file is flushed / updated (in seconds). The file will always be flushed
        once
    """

    # ATTRIBUTES #

    # FILE HANDLING

    # Handle of the internal file and CSV writer
    _file: IO
    _csv_writer: IO

    # File name and file path
    _file_name: str
    _file_path: str

    # FILE FLUSHING

    # Time of creation / last flush
    _last_update_time: float

    # Frequency of flushing (in seconds)
    _flush_frequency: int

    # CONSTRUCTOR #
    def __init__(self, results_path, input_name, flush_frequency):

        # Check if the path exists and, if necessary, create the folders
        if not os.path.exists(results_path):
            Path(results_path).mkdir(parents=True, exist_ok=True)

        # Store the current time and the flush frequency
        self._last_update_time = time.time()
        self._flush_frequency = flush_frequency

        # Create the input file name
        # <input_name>_<time>.csv
        self._file_name = "{}_{}.csv".format(input_name, self._last_update_time)
        # Create the path for the actual file
        self._file_path = os.path.join(results_path, input_name)

        # Create the file and store the handles
        self._create_results_file(self._file_path)

    # TODO FINISH METHODS

    # FILE AND PATH MANAGEMENT
    def _create_results_file(self, file_path):
        """
        Given a path and a file name, creates a file handle and its associated CSV writer.

        Parameters
        ----------
        file_path: str
            Path of the results file
        """

        # Open the file in write-only mode
        self._file = open(file_path, "wt", newline="")

        # Open the CSV handler for said file
        self._csv_writer = csv.writer(self._file, delimiter=",")

    def close_results_file(self):
        raise NotImplementedError

    # LOG MANAGEMENT
    def write_empty_line(self):
        raise NotImplementedError

    def write_header(self):
        raise NotImplementedError

    def write_column_titles(self):
        raise NotImplementedError

    def write_data(self):
        raise NotImplementedError