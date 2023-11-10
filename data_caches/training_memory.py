# BAYESIAN NETWORK - NN GS SPEEDUP #
# Developed by Luna Jimenez Fernandez
# Based on the work of Wenfeng Zhang et al.

# IMPORTS #

from typing import List

import torch
import numpy as np


class TrainingMemory:
    """
    Class containing a memory used to store (Tensor, score) pairs where:
        - Tensor represents the input to be used for the neural network
        - Score represents the expected output for said input

    This memory is used to train the neural network during the Bayesian Network
    building process, being wiped after each episode.

    In addition, this memory contains auxiliary methods to sample and clear the memory.
    """

    # ATTRIBUTES #

    # Tensors stored within the memory
    _tensor_memory: List[torch.Tensor]
    # Scores stored within the memory
    _score_memory: List[float]

    # Both lists are synchronized, and separate lists are used
    # to speed up the sampling process

    # CONSTRUCTOR #

    def __init__(self):
        self.wipe_memory()

    # INSERTION AND SAMPLING METHODS #

    def insert_memory(self, tensor, score):
        """
        Inserts a (tensor, expected score) pair into the training memory

        Parameters
        ----------
        tensor: torch.Tensor
            Tensor representing the input of the neural network
        score: float
            Expected score for the tensor. Usually a BDeU or local BDeU score.
        """

        self._tensor_memory.append(tensor)
        self._score_memory.append(score)

    def sample_memory(self, chunk_size):
        """
        Returns the contents of the memory as lists of Tensor chunks of the specified size,
        prepared for Neural Network training.

        The lists contain:
            - A Tensor with all required Input tensors
            - A Tensor with all required Output float values

        The memory is shuffled before sampling to avoid correlations during training.

        Parameters
        ----------
        chunk_size: int
            Size of each chunk / how many inputs or outputs are in each tensor.
            Used for parallelization.

        Returns
        -------
        tuple[list[torch.Tensor], list[torch.Tensor]]
        """

        # SHUFFLING

        # Convert both lists into Numpy array for easier work
        tensor_array = np.array(self._tensor_memory)
        score_array = np.array(self._score_memory)

        # Generate random indices for shuffling
        shuffle_indices = np.arange(len(tensor_array))
        np.random.shuffle(shuffle_indices)

        # Shuffle both arrays
        tensor_array = tensor_array[shuffle_indices]
        score_array = score_array[shuffle_indices]

        # CHUNKING

        # Prepare lists to store the chunks
        tensor_list = []
        score_list = []

        # Compute the total amount of chunks and process each chunk separately
        total_chunk_amount = len(tensor_array) // chunk_size
        chunk_remainder = len(tensor_array) % chunk_size

        # Process the full chunks and insert them
        for i in range(total_chunk_amount):

            # Tensors
            tensor_chunk = [tensor for tensor in tensor_array[chunk_size*i:chunk_size*(i+1)]]
            tensor_chunk = torch.stack(tensor_chunk)
            tensor_list.append(tensor_chunk)

            # Scores
            score_chunk = [score for score in score_array[chunk_size*i:chunk_size*(i+1)]]
            score_chunk = torch.tensor(score_chunk)
            score_list.append(score_chunk)

        # If necessary, process the chunk remainders
        if chunk_remainder > 0:

            # Tensors
            tensor_chunk = [tensor for tensor in tensor_array[chunk_size * total_chunk_amount:chunk_size * total_chunk_amount + chunk_remainder]]
            tensor_chunk = torch.stack(tensor_chunk)
            tensor_list.append(tensor_chunk)

            # Scores
            score_chunk = [score for score in score_array[chunk_size * total_chunk_amount:chunk_size * total_chunk_amount + chunk_remainder]]
            score_chunk = torch.tensor(score_chunk)
            score_list.append(score_chunk)

        # Return the lists of prepared and sampled tensors
        return tensor_list, score_list

    # AUXILIARY METHODS #

    def wipe_memory(self):
        """
        Wipes the memory, emptying it and preparing it for a new episode.
        """

        # Initialize both lists (empty)
        self._tensor_memory = []
        self._score_memory = []
