# BAYESIAN NETWORK - NN GS SPEEDUP #
# Developed by Luna Jimenez Fernandez
# Based on the work of Wenfeng Zhang et al.

# IMPORTS #
import os.path
from pathlib import Path
import csv

import time

from typing import IO, Any


class ResultsLogger:
    """
    "ResultsLogger" provides a wrapper for a CSV writer, containing all necessary methods to create and handle
    a CSV results file - including headers, column names, data writing and commentaries for final results.

    "ResultsLogger" also includes methods for:

        - Writing comment blocks with variable number of details.
        - Writing headers and iteration data.

    The resulting results file will have "<input_name>_<creation time>.csv" as a name.

    Parameters
    ----------
    results_path: str
        Location of the final CSV results file. Note that this path should NOT include the file name.
    output_name: str
        Name of the output data to use. Usually, the name of either the data file or the given dataset.
    flush_frequency: int
        How often the results file is flushed / updated (in seconds). The file will always be flushed
        once
    """

    # ATTRIBUTES #

    # FILE HANDLING

    # Handle of the internal file and CSV writer
    _file: IO
    _csv_writer: Any

    # File name and file path
    file_name: str
    file_path: str

    # FILE FLUSHING

    # Time of creation / last flush
    last_update_time: float

    # Frequency of flushing (in seconds)
    flush_frequency: int

    # CONSTRUCTOR #
    def __init__(self, results_path, output_name, flush_frequency):

        # Check if the path exists and, if necessary, create the folders
        if not os.path.exists(results_path):
            Path(results_path).mkdir(parents=True, exist_ok=True)

        # Store the current time and the flush frequency
        self.last_update_time = time.time()
        self.flush_frequency = flush_frequency

        # Create the input file name
        # <output_name>_<time>.csv
        self.file_name = "{}_{}.csv".format(output_name, self.last_update_time)
        # Create the path for the actual file
        self.file_path = os.path.join(results_path, self.file_name)

        # Create the file and store the handles
        self._create_results_file(self.file_path)

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
        """
        Closes both file handles orderly to release system resources
        """

        self._csv_writer.close()
        self._file.close()

    # INTERNAL LOG MANAGEMENT
    # NOTE - Since headers and final results do not have constant formats (since the data shown might depend on the
    # specified parameters), both header and results should be printed using comments.

    def write_line(self, line, printed=False, remove_leading_characters=True):
        """
        Directly writes a line into the file by bypassing the CSV writer. If specified, the line is also printed
        as-is into the console.

        Can be used to:
        - Write comments (by starting the line with #)
        - Write empty lines (by passing "\n" as a line)

        Parameters
        ----------
        line: str
            Line to write into the file
        printed: bool
            If True, line is also printed into the console.
        remove_leading_characters: bool
            If True, all leading characters ("" and white space) will be removed.
        """

        self._file.write(line)
        if printed:

            # Preprocess the string
            if remove_leading_characters:
                line = line.strip("# ")

            print(line, end="")

    def write_data_row(self, data):
        """
        Writes a data row according to CSV standards and, if enough time has passed
        (as specified in the constructor), flushes the file

        This row can be either:
        - Column headers
        - Value of an iteration

        Parameters
        ----------
        data: list
            List of data to write
        """

        # Write the row of data
        self._csv_writer.writerow(data)

        # Check the time that has passed and, if necessary, flush the file
        if time.time() - self.last_update_time > self.flush_frequency:

            # Flush the file
            self._file.flush()

            # Update the timer
            self.last_update_time = time.time()

    # LOG AND PRINTING METHODS
    # NOTE - These methods will both write to the log and print to the screen according to the specified verbosity.

    def write_comment_block(self, block_title, data_dictionary, verbosity):
        """
        Writes and prints the information specified within data_dictionary as a "comment block" (data
        that is not part of an iteration, and that should not be read from a CSV file).

        data_dictionary is expected to have the following structure:

        {
            ("<data_name>", tab, verbosity): (value, unit)
        }

        Where:

            - "data_name": Name (as an elaborated string) of the variable.
            - "tab": Boolean, if true, the value is tabulated.
            - "verbosity": Expected verbosity of this value. If the current verbosity is lower, it will not
                           be printed to the screen (BUT IT WILL STILL BE LOGGED)
            - "value": Actual value to be printed
            - "unit": If not None, unit to specify (f.ex. secs, %...)

        Parameters
        ----------
        block_title: str
            Header of the comment block
        data_dictionary: dict
            Dictionary with the data, using the specified name
        verbosity: int
            Current verbosity of the terminal. Verbosity will be checked against the data to see what information
            needs to be printed. If verbosity is 0 or lower, no console logging will be done at all.
        """

        # Block start and block title
        self.write_line("########################################\n")
        self.write_line(f"# {block_title}\n\n", printed=(verbosity > 0))

        # Dictionary values
        for (data_name, data_tab, data_verbosity), (data_value, data_unit) in data_dictionary.items():

            # Check whether the verbosity is valid
            valid_verbosity = (verbosity > 0) and (verbosity >= data_verbosity)

            # A different string is prepared depending on whether a tab is needed or not
            if not data_tab:
                self.write_line(f"# - {data_name}: {data_value} {data_unit}\n",
                                printed=valid_verbosity)
            else:
                self.write_line(f"#\t * {data_name}: {data_value} {data_unit}\n",
                                printed=valid_verbosity)

        # Block end
        self.write_line("\n########################################\n\n")

# TODO METHODS FOR DATA WRITING AND DAG WRITING