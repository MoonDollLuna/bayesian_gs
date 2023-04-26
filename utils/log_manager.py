# BAYESIAN NETWORK - NN GS SPEEDUP #
# Developed by Luna Jimenez Fernandez
# Based on the work of Wenfeng Zhang et al.

# TODO Complete
# - TODO Specify path
# IMPORTS #

class LogManager:

    # TODO FINISH METHODS

    # FILE MANAGEMENT
    def create_file(self):
        raise NotImplementedError

    def close_file(self):
        raise NotImplementedError

    # LOG MANAGEMENT
    def write_header(self):
        raise NotImplementedError

    def write_column_titles(self):
        raise NotImplementedError

    def write_data(self):
        raise NotImplementedError