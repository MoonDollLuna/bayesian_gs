# BAYESIAN NETWORK - NN GS SPEEDUP #
# Developed by Luna Jimenez Fernandez
# Based on the work of Wenfeng Zhang et al.

# IMPORTS #

import random

from collections import deque
import torch


class TensorMemory:
    """
    Memory storing the tensors used to train the neural networks, along with the expected scores
    for each tensor. These are stored in (tensor, score) format, and used for training.

    This memory may optionally have a maximum size. If specified, once the memory is full the oldest
    (tensor, score) pairs will be removed in FIFO order.

    This class implements methods to insert, extract and sample the memory directly
    or in randomized chunks.

    Parameters
    ----------
    max_size : int, optional
        Maximum size of the memory
    """

    # ATTRIBUTES #

    # Queue used to store the (tensor, score) pairs
    # This queue can have a maximum size specified during construction
    _memory: deque

    # Maximum size of the deque
    max_size: int or None

    # CONSTRUCTOR #

    def __init__(self, max_size=None):

        # Initialize the deque
        self._memory = deque(maxlen=max_size)

        # Store the maximum size
        self.max_size = max_size

    # INSERTION METHODS #

    def insert_tensor(self, tensor, score):
        """
        Inserts a (tensor, expected score) pair into the Tensor Memory.

        Parameters
        ----------
        tensor: torch.Tensor
            Tensor representing the input of the neural network
        score: float
            Expected score for the tensor. Usually a BDeU or local BDeU score.
        """

        self._memory.append((tensor, score))

    # SAMPLING METHODS
    # TODO

    def sample_chunk(self, chunk_size):
        pass

    def sample_multiple_chunks(self, chunk_amount, chunk_size):
        pass

    def sample_memory_in_chunks(self, chunk_size):
        pass







