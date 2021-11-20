import sys
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
        Instantiate variables for matrix factorization, i.e. M = A * B + C.
        :param M <torch.Tensor|numpy.ndarray>:  Input 2-D matrix to factorize into A and B (with bias C).
        :param dim <int>:                       Hidden / latent dimension of matrix factorization.
        :param bias <bool>:                     Utilize bias in factorization. Set to False to exclude affine bias C.
        :param seed <int>:                      Random seed fixture for reproducibility.
        """
        super().__init__()

        # Validate matrix.
        if len(M.shape) != 2 or dim <= 0:
            print(
                f"[MatrixInputError] MatrixFactorize() only supports 2-D matrices. " +
                f"Moreover, the hidden dimension is necessarily > 0.",
                file=sys.stderr, flush=True
            )
            return None

        # Class Parameters
        self.M = torch.Tensor(M)
        self.x, self.y = self.M.shape
        self.dim = dim
        self.bias = bias

        # Fix random seed.
        if seed is not None:
            torch_gen = torch.manual_seed(seed)

        # Matrix Factors + Affine Bias
        self.A = Parameter(torch.empty((self.x, self.dim)), requires_grad=True)
        self.B = Parameter(torch.empty((self.dim, self.y)), requires_grad=True)
        self.C = Parameter(torch.empty((self.x, self.y)), requires_grad=True)
        init.xavier_uniform_(self.A)
        init.xavier_uniform_(self.B)
        init.xavier_uniform_(self.C)

        # Optimizer
        self.optimizer = AdamW(self.parameters())

    def forward(self):
        """
        Compute element of matrix from factorization.
        """
        
        # Compute product matrix or recommendation.
        output = torch.matmul(self.A, self.B)
        if self.bias:
            # Include affine bias.
            output += self.C
        return output

    def fit(self):
        """
        Train the factorization.
        """

        pass