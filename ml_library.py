import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import *
from torch.optim import AdamW
import numpy as np
from typing import Union

class MatrixFactorize(Module):
    """
    Matrix Factorization Model for Numerical Factorization and Recommender Systems
    """

    def __init__(self, M: Union[torch.Tensor, np.ndarray], dim, bias=True, seed=None):
        """
        Instantiate variables for matrix factorization, i.e. M = A * B.
        :param M <torch.Tensor|numpy.ndarray>:  Input 2-D matrix to factorize into A and B.
        :param dim <int>:                       Hidden / latent dimension of matrix factorization.
        :param bias <bool>:                     Utilize bias in factorization. Set to False for pure linear factorization.
        :param seed <int>:                      Random seed fixture for reproducibility.
        """
        super().__init__()

        # Parameters
        self.x, self.y, *_ = M.shape
        self.dim = dim
        self.bias = bias

        # Fix random seed.
        if seed is not None:
            torch.manual_seed(seed)

        # Matrix Factors
        self.A = Embedding(self.x, dim)
        self.B = Embedding(self.y, dim)

        # Biases
        self.c1 = Embedding(self.x, 1)
        self.c2 = Embedding(self.y, 1)

        # 

    def matrix_factors(self):
        """
        Output matrix factorization, i.e. self.A and self.B.
        """

        return {
            'A': self.A(torch.LongTensor([range(self.x)])).flatten(end_dim=1),
            'B': self.B(torch.LongTensor([range(self.y)])).flatten(end_dim=1).t(),
            'A_c': self.c1(torch.LongTensor([range(self.x)])).flatten(end_dim=1),
            'B_c': self.c2(torch.LongTensor([range(self.y)])).flatten(end_dim=1)
        }

    def forward(self, x, y):
        """
        Compute element of matrix from factorization.
        :param x <int>:     X-index of the matrix.
        :param y <int>:     Y-index of the matrix.
        """

        # Lookup weights and biases.
        A_x = self.A(torch.LongTensor([x]))
        c_x = self.c1(torch.LongTensor([x]))
        B_y = self.B(torch.LongTensor([y]))
        c_y = self.c2(torch.LongTensor([y]))

        # Estimate matrix product or recommendation.
        output = torch.matmul(A_x, B_y.t())
        if self.bias:
            # Include affine bias.
            output += c_x + c_y
        return torch.flatten(output)

    def fit(self):
        """
        Train the factorization.
        """

        pass