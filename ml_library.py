import sys
import math
import numpy as np
import torch
from torch.nn import *
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from typing import Union

class MatrixFactorize(Module):
    """
    Matrix Factorization Model for Numerical Factorization and Recommender Systems
    """

    def __init__(self, M: Union[torch.Tensor, np.ndarray], dim, mask: Union[torch.Tensor, np.ndarray] = None, bias=True, seed=None):
        """
        Instantiate variables for matrix factorization, i.e. M = A * B + C.
        :param M <torch.Tensor|numpy.ndarray>:      Input 2-D matrix to factorize into A and B (with bias C).
        :param dim <int>:                           Hidden / latent dimension of matrix factorization.
        :param mask <torch.Tensor|numpy.ndarray>:   Mask matrix with shape equivalent to M. 
                                                    Non-zero implies True, while zero implies False.
        :param bias <bool>:                         Utilize bias in factorization. Set to False to exclude affine bias C.
        :param seed <int>:                          Random seed fixture for reproducibility.
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
        self.mask = torch.ones(self.M.shape)
        if mask is not None and mask.shape == self.M.shape:
            # Construct Boolean mask Tensor.
            self.mask = torch.where(torch.Tensor(mask) != 0, 1, 0)

        # Fix random seed.
        self.torch_gen = None
        if seed is not None:
            self.torch_gen = torch.manual_seed(seed)

        # Matrix Factors + Affine Bias
        self.A = Parameter(torch.empty((self.x, self.dim)), requires_grad=True)
        self.B = Parameter(torch.empty((self.dim, self.y)), requires_grad=True)
        self.C = Parameter(torch.empty((self.x, self.y)), requires_grad=True)
        init.xavier_uniform_(self.A)
        init.xavier_uniform_(self.B)
        init.xavier_uniform_(self.C)

        # Track training loss.
        self.loss = float('inf')

    def __repr__(self):
        """
        Output matrix factorization and approximation matrix of M.
        """

        # Avoid scientific notation.
        np.set_printoptions(suppress=True)

        # Collect information.
        output_buffer = f"\nMatrix Factorization Output (M = A * B + C)\n"
        output_buffer += f"\nTarget Matrix:\n\n{self.M.numpy()}\n"
        output_buffer += f"\nMask Matrix:\n\n{self.mask.numpy()}\n"
        output_buffer += f"\nApprox. Matrix:\n\n{self().detach().numpy().round(2)}\n"
        output_buffer += f"\nMatrix A:\n\n{self.A.detach().numpy().round(2)}\n"
        output_buffer += f"\nMatrix B:\n\n{self.B.detach().numpy().round(2)}\n"
        output_buffer += f"\nMatrix C:\n\n{self.C.detach().numpy().round(2)}\n"
        output_buffer += f"\nRegression Loss: {self.loss}\n"

        # Reset notation.
        np.set_printoptions(suppress=False)

        return output_buffer

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

    def fit(self, cycles=1000, lr=8e-4, regularize=1e-2, patience=3, verbose=False):
        """
        Train the factorization.
        :param cycles <int>:        Number of gradient descent cycles.
        :param lr <float>:          Learning rate. Re-calibrated to order of values in matrix M.
        :param regularize <float>:  Weight decay lambda for regularization in AdamW.
        :param patience <int>:      Number of cycles of convergence before termination.
        :param verbose <bool>:      Output training progress information.
        """

        # Instantiate optimizer. Tune learning rate to order of magnitude of matrix M.
        lr_calibrate = lr * self.M.sum() / torch.numel(self.M)
        optimizer = AdamW(
            self.parameters(),
            lr=lr_calibrate,
            weight_decay=regularize
        )

        # Train factorization.
        factor_opt = dict(self.state_dict())
        timer = 0
        for i in range(cycles):

            # Clear gradients in optimizer.
            self.zero_grad()

            # Compute loss of matrix factorization.
            loss = torch.linalg.norm(
                (self() - self.M) * self.mask
            ) ** 2

            # Back-propagate.
            loss.backward()

            # Apply gradients.
            optimizer.step()

            # Validate training.
            if i % 100 == 0 and verbose:
                print(
                    f"Training Cycle: {i+1} / {cycles} | Factorization Loss: {loss.item()}",
                    file=sys.stdout, flush=True
                )
            if loss.item() < self.loss:
                # Update optimal factorization.
                factor_opt = dict(self.state_dict())
                # Update optimal loss.
                self.loss = loss.item()
                # Reset patience timer.
                timer = 0
            else:
                # Increment patience timer.
                timer += 1
                if timer > patience:
                    # Model convergence. Revert to optimal factorization. Terminate training.
                    self.load_state_dict(factor_opt)
                    break


class FactorizationMachine(Module):
    """
    [Factorization Machine Recommendation Model]
    Learns latent space features to characterize similarity of dataset features
    to compute a recommendation as a function of dataset features. Dataset
    features can be mixed / hybrid such that you can combine information
    on both the recommended object and the recommendation target to generate
    an informed similarity or recommendation / ranking metric.
    """

    def __init__(self, data_dim, hidden_dim=25, seed=None) -> None:
        """
        Instantiate class attributes for FM. Constructs a feature similarity matrix
        F of shape (x_features, hidden_dim) to learn implicit representations of
        all trainable features in the data for recommendation or ranking.
        :param data_dim <int>:      Number of features to learn from in the dataset.
        :param hidden_dim <int>:    Dimension of the latent space of features.
        :param seed <int>:          Random seed fixture for reproducibility.
        """
        super().__init__()

        # Parameters
        self.input_dim = data_dim       # X
        self.hidden_dim = hidden_dim    # H

        # Fix random seed.
        self.torch_gen = None
        if seed is not None:
            self.torch_gen = torch.manual_seed(seed)

        """ Matrix Factorization """

        # Feature Similarity Matrix
        self.F = Parameter(
            torch.empty((self.input_dim, self.hidden_dim)),
            requires_grad=True
        )
        init.xavier_uniform_(self.F)

        """ Linear Regression """

        # Feature Weight Vector and Bias
        self.V = Parameter(
            torch.empty((self.input_dim, 1)),
            requires_grad=True
        )
        init.xavier_uniform_(self.V)
        self.bias = Parameter(
            torch.zeros(1),
            requires_grad=True
        )

        """ Gaussian Regression """

        self.gaussian_dist = Linear(
            in_features=self.hidden_dim,
            out_features=2
        )

    def forward(self, x: torch.Tensor):
        """
        Compute FactorizationMachine(x). Returns a mean and standard deviation for the recommendation.
        :param x <torch.Tensor>:    Factorization machine input Tensor of shape (N, input_dim).
        """

        # Compute square of sum and sum of squares.
        # (N, X) * (X, H) -> (N, H)
        sq_sm = torch.matmul(x, self.F) ** 2
        sm_sq = torch.matmul(x ** 2, self.F ** 2)

        # Compute linear regression model.
        # (N, H) * (H, 1) -> (N, 1)
        lin_reg = torch.matmul(x, self.V)

        # Compute latent feature matrix of shape (N, H).
        latent = self.bias + lin_reg + 0.5 * sq_sm - sm_sq

        # Fit Gaussian distribution to recommendation.
        # (N, H) -> (N, 2)
        output = self.gaussian_dist(latent)

        # Output recommendation / ranking score distribution, i.e. FM(x).
        return output[:, 0], torch.abs(output[:, 1])

    def fit(
        self, 
        X: Union[torch.Tensor, np.ndarray], 
        Y: Union[torch.Tensor, np.ndarray], 
        cycles=100, 
        lr=2e-3, 
        batch_frac=0.01, 
        regularize=1e-2, 
        patience=3, 
        verbose=False
    ):
        """
        Train the Factorization Machine.
        :param X <torch.Tensor>:    Input training data features of shape (N, input_dim).
        :param Y <torch.Tensor>:    Target training data class / score vector of shape (N, 1).
        :param cycles <int>:        Number of gradient descent cycles.
        :param lr <float>:          Learning rate. Re-calibrated to order of values in matrix M.
        :param batch_frac <float>:  Fraction of the dataset to set as the batch size.
        :param regularize <float>:  Weight decay lambda for regularization in AdamW.
        :param patience <int>:      Number of cycles of convergence before termination.
        :param verbose <bool>:      Output training progress information.
        """

        # Validate arguments.
        if any([
            len(X.shape) != 2,
            len(Y.shape) != 2,
            X.shape[1] != self.input_dim,
            Y.shape[1] != 1,
            cycles <= 0,
            lr <= 0,
            batch_frac <= 0,
            regularize < 0
        ]):
            print(
                f"[FactorizationMachine.fit()] Improper training argument(s) to fit(). Aborting... \n" +
                f"""FactorizationMachine.fit() Requirements:
                - X is a (N, x_features) Tensor and Y is a (N, 1) Tensor.
                - cycles > 0
                - lr > 0
                - batch_frac > 0
                - regularize >= 0
                """,
                file=sys.stderr, flush=True
            )
            return

        # Convert to torch.Tensor.
        N = X.shape[0]
        if not torch.is_tensor(X):
            X = torch.Tensor(X)
        if not torch.is_tensor(Y):
            Y = torch.Tensor(Y)

        # Instantiate optimizer.
        optimizer = AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=regularize
        )

        # Train Factorization Machine.
        model_opt = dict(self.state_dict())
        loss_opt = float('inf')
        timer = 0
        for i in range(cycles):

            # Train on batches of data.
            for _ in range(math.ceil(1 / batch_frac)):

                # Generate random batches by index.
                rand_idx = torch.randint(
                    N,
                    size=(math.ceil(batch_frac * N),),
                    generator=self.torch_gen
                )

                # Extract batch.
                X_batch = X[rand_idx]
                Y_batch = Y[rand_idx]

                # Clear gradients in optimizer.
                self.zero_grad()

                # Compute model prediction mean and deviation.
                Y_mu, Y_sigma = self(X_batch)

                # Compute Gaussian distributional loss.
                loss = GaussianNLLLoss()(Y_mu, Y_batch, Y_sigma)

                # Back-propagate.
                loss.sum().backward()

                # Apply gradients.
                optimizer.step()

            # Validate training.
            if i % math.ceil(cycles / 5) == 0 and verbose:
                print(
                    f"Training Cycle: {i+1} / {cycles} | Recommendation Loss: {loss.item()}",
                    file=sys.stdout, flush=True
                )
            if loss.sum().item() < loss_opt:
                # Update optimal factorization.
                model_opt = dict(self.state_dict())
                # Update optimal loss.
                loss_opt = loss.sum().item()
                # Reset patience timer.
                timer = 0
            else:
                # Increment patience timer.
                timer += 1
                if timer > patience:
                    # Model convergence. Revert to optimal factorization. Terminate training.
                    self.load_state_dict(model_opt)
                    break