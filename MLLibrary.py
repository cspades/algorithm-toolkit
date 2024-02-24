"""
Machine Learning Library (Python v3.12.2+ / PyTorch 2.2.1+cu121)
Implemented by Cory Ye
For personal educational review.
"""
import sys
import math
import numpy as np
import torch
from torch.nn import *
from torch.optim import AdamW
from typing import Union

class GaussianMixture:
    """
    Unsupervisedly fits a mixture of Gaussian distributions onto a dataset
    via the expectation maximization algorithm (EM).

    For more information, refer to the derivation:
    https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm#Gaussian_mixture
    """

    def __init__(self, data: Union[torch.Tensor, np.ndarray], k: int, delta=1e-2, seed: int = None):
        """
        Instantiate variables for EM.
        :param data <torch.Tensor|numpy.ndarray>:   Input (N,d) tensor / n-dimensional array of data to fit a GaussianMixture.
        :param k <int>:                             Number of Gaussian distributions to fit to the data via EM.
        :param delta <float>:                       Delta on the model parameters that terminates training.
        :param seed <int>:                          Random seed to fix the initialization of the Gaussians.
        """
        # Import data as torch.Tensor.
        self.data = torch.Tensor(data)
        self.N, self.D = self.data.shape
        self.K = k

        # Algorithm Parameters
        self.delta = delta
        self.modelDelta = float('inf')
        self.eps = 1e-2
        self.bound = self.data.abs().max().item()
        self.noise = 1e-5

        # Fix random seed.
        self.torch_gen = None
        if seed is not None:
            self.torch_gen = torch.manual_seed(seed)

        # Initialize K-Means to randomly chosen points in self.data.
        # Initializing means to existing datapoints drastically
        # decreases the probability of non-convergence.
        indexShuffle = torch.randperm(self.N)
        self.mu = self.data[indexShuffle[:self.K]]

        # Initialize K-Covariances. Must be positive-definite,
        # so we compute the Gram matrix, i.e. A^T * A.
        # Initiatizing covariances to the square of twice the 
        # magnitude of the datapoint accelerates convergence
        # and reduces the probability of non-convergence.
        sigmaInit = torch.zeros(self.K, self.D, self.D)
        init.xavier_uniform_(sigmaInit, gain=math.pow(2*self.bound, 2))
        self.sigma = self.gramMatrix(sigmaInit)

        # Initialize class membership probabilities and class prior probabilities to 1/K.
        self.classMemberProbs = torch.ones(self.N, self.K) / self.K
        self.classPriorProbs = torch.ones(self.K) / self.K

    def __repr__(self):
        """
        Print the mean and covariance matrices of all Gaussians.
        """
        # Avoid scientific notation.
        np.set_printoptions(suppress=True)

        outputRepr = "\n[> GaussianMixture Model Summary <]\n"
        for j in range(self.K):
            outputRepr += f"\n---------------------------------------\n"
            outputRepr += f"\n{{Gaussian Distribution Class {j}}}\n"
            outputRepr += f"\nMean:\n{self.mu[j].numpy().round(2)}\n"
            outputRepr += f"\nCovariance:\n{self.sigma[j].numpy().round(2)}\n"
            classMemberData = self.data[self.classMemberProbs.numpy().argmax(axis=1) == j]
            outputRepr += f"\nTraining Class Members:\n{classMemberData.numpy().round(2)}\n"

        # Reset notation.
        np.set_printoptions(suppress=False)

        # Print.
        return outputRepr
    
    def fit(self):
        """
        Apply the expectation maximization unsupervised clustering algorithm (EM)
        to optimize the mean and covariance of Gaussian distributions
        such that the probability that all datapoints were sampled from
        the mixture of Gaussian distributions is maximized.

        Intuitively, alternate between updating the posterior distribution
        (i.e. class membership probability) and optimizing the model parameters
        (i.e. mean, covariance, priors) of each Gaussian distribution to converge
        to an optimal model fit of the sample dataset.
        """
        # Track the absolute difference in model parameters.
        prevMu = self.mu
        prevSigma = self.sigma
        prevPrior = self.classPriorProbs
        while self.modelDelta > self.delta:
            """
            Expectation Step (E) - Compute the class membership
            probabilities for each datapoint in our training dataset. 
            """
            # Compute class membership / sampling probabilities.
            self.classMemberProbs = self.infer(self.data)
            
            """
            Maximization Step (M) - Fit the mixture of Gaussian
            distributions to the datapoints weighted by the
            probability that the datapoint was sampled from
            the distribution. Update the model delta.
            """
            # Derive class prior probabilities of shape (K).
            self.classPriorProbs = self.classMemberProbs.mean(dim=0)

            # Optimize expected / average class mean parameters.
            # (K, D) = (K, N) x (N, D) / (K, 1)
            self.mu = torch.div(
                torch.matmul(torch.transpose(self.classMemberProbs, 0, 1), self.data),
                self.classMemberProbs.sum(dim=0).view(self.K, 1)
            )

            # Optimize expected / average class covariance parameters.
            data_reshaped = self.data.view(1, self.N, self.D, 1).expand(self.K, -1, -1, 1)
            mu_reshaped = self.mu.view(self.K, 1, self.D, 1).expand(-1, self.N, -1, 1)
            # Reshape to (K, N, D, 1) to compute covariance of shape (D, D) = (D, 1) x (1, D).
            deviation = data_reshaped - mu_reshaped
            # (K, N, D, D) = (K, N, D, 1) x (K, N, 1, D)
            covariance = torch.matmul(deviation, torch.transpose(deviation, 2, 3))
            # Compute expected / average covariance.
            # (K, D, D) = [(K, N, D, D) * (K, N, 1, 1)].sum(N) / (K, 1, 1)
            self.sigma = torch.div(
                torch.mul(covariance, torch.transpose(self.classMemberProbs, 0, 1).view(self.K, self.N, 1, 1)).sum(dim=1),
                self.classMemberProbs.sum(dim=0).view(self.K, 1, 1)
            )

            # Compute model delta.
            self.modelDelta = sum([
                # Gaussian Mean Convergence
                torch.norm(self.mu - prevMu).item() / torch.norm(prevMu).item(),
                # Gaussian Covariance Convergence
                torch.norm(self.sigma - prevSigma).item() / torch.norm(prevSigma).item(),
                # Distributional Convergence
                torch.norm(self.classPriorProbs - prevPrior).item() / torch.norm(prevPrior).item()
            ])
            # Track previous model parameters.
            prevMu = self.mu
            prevSigma = self.sigma
            prevPrior = self.classPriorProbs
        
        # Print training summary.
        print(self)
        print(f"Termination Delta: {self.modelDelta}")

    def infer(self, x: Union[torch.Tensor, np.ndarray]):
        """
        Infer the class from which x was sampled, i.e. Expectation in Expectation-Maximization (EM).
        Returns a Tensor of shape (M, K) containing class sampling / membership probabilities.
        :param x <torch.Tensor|np.ndarray>:     Input tensor of shape (M, D).
        """
        # Compute relative class sampling probabilities for all classes.
        # P[Data X is sampled from class J. | Class J is modeled by (Mu[j], Sigma[j]).] * P[Data has class J.]
        # (M, K) = (M, K) * (K)
        classSampleProbs = torch.mul(self.gaussianPdf(x, self.mu, self.sigma), self.classPriorProbs)

        # Compute the cumulative sampling probability, i.e. P[Data X is sampled from Model.]
        # (M, 1) = (M, K).sum(K), reshaped for right-to-left broadcasting compatibility with (M, K).
        M = x.shape[0]
        combSampleProb = classSampleProbs.sum(dim=1).view(M, 1)

        # Divide relative class sampling probability by cumulative sampling probability
        # to derive a class membership conditional probability, i.e.
        # P[Data X is sampled from class J. | Data X is sampled from Model.] =
        # P[Data X is sampled from class J. | Class J is modeled by (Mu[j], Sigma[j]).] * P[Data X has class J.] / P[Data X is sampled from Model.]
        # (M, K) = (M, K) / (M, 1)
        return torch.div(classSampleProbs, combSampleProb)

    def gaussianPdf(self, x: Union[torch.Tensor, np.ndarray], mu: Union[torch.Tensor, np.ndarray], sigma: Union[torch.Tensor, np.ndarray]):
        """
        Compute the probability density function for the multivariate Gaussian distribution.
        Utilized relatively to estimate the posterior probability given that a datapoint x is
        sampled from the Gaussian distribution of mean mu and covariance sigma.
        :param x <torch.Tensor|np.ndarray>:         Input tensor of shape (M, D).
        :param mu <torch.Tensor|np.ndarray>:        Mean tensor of shape (K, D).
        :param sigma <torch.Tensor|np.ndarray>:     Covariance tensor of shape (K, D, D).
        :return <torch.Tensor|np.ndarray>:          Probability matrix of shape (M, K) for each sample M and each distribution K.
        """
        # Compute probability density function for the Gaussian.
        return torch.div(
            # (M, K)
            self._perturbedClamp(torch.exp(-0.5 * self.mahalanobisDistance(x, mu, sigma)), min=self.eps, max=self.bound, noise=self.noise),
            # (1, K)
            torch.sqrt(pow(2 * math.pi, self.D) * torch.det(sigma).view(1, self.K) + self.eps)
        )
    
    def mahalanobisDistance(self, x: Union[torch.Tensor, np.ndarray], mu: Union[torch.Tensor, np.ndarray], sigma: Union[torch.Tensor, np.ndarray]):
        """
        Compute the (squared) Mahalanobis distance from the datapoint X to the distribution parametrized by (Mu, Sigma).
        :param x <torch.Tensor|np.ndarray>:         Input tensor of shape (M, D).
        :param mu <torch.Tensor|np.ndarray>:        Mean tensor of shape (K, D).
        :param sigma <torch.Tensor|np.ndarray>:     Covariance tensor of shape (K, D, D).
        :return <torch.Tensor|np.ndarray>:          Distance matrix of shape (M, K) for each sample M and each distribution K.
        """
        # Reshape both x and mu to (M, K, D, 1). Reshape sigma to (M, K, D, D).
        M = x.shape[0]
        x_reshaped = x.view(M, 1, self.D, 1).expand(-1, self.K, -1, 1)
        mu_reshaped = mu.view(1, self.K, self.D, 1).expand(M, -1, -1, 1)
        sigma_reshaped = sigma.view(1, self.K, self.D, self.D).expand(M, -1, -1, -1)
        # Compute x - mu.
        deviation = x_reshaped - mu_reshaped
        # Compute the squared Mahalanobis distance.
        # (M, K, 1, 1) = (M, K, 1, D) x (M, K, D, D) x (M, K, D, 1)
        mahalanobis = torch.matmul(
            torch.transpose(deviation, 2, 3),
            torch.matmul(torch.linalg.inv(sigma_reshaped), deviation)
        )
        # Reshape to (M, K).
        return mahalanobis.view(M, self.K)
    
    def gramMatrix(self, tensor: Union[torch.Tensor, np.ndarray]):
        """
        Compute a normalized Gram matrix, i.e. A^T A / sqrt(|A|).
        """
        # Compute rank of tensor.
        tensorRank = len(tensor.shape)
        if tensorRank < 2 or tensor.shape[-1] != tensor.shape[-2]:
            raise ValueError("Cannot construct Gram Matrix for Tensor with non-square tail dimensions, or Tensor(s) of rank < 2.")
        # Compute Gram Matrix.
        gram = torch.matmul(torch.transpose(tensor, tensorRank-2, tensorRank-1), tensor)
        # Normalize.
        gramAbsSqrt = torch.sqrt(gram.abs())
        return gram.sign() * gramAbsSqrt
    
    def _perturbedClamp(self, tensor: Union[torch.Tensor, np.ndarray], min: float = None, max: float = None, noise: float = None):
        """
        Bound the magnitude / absolute value of tensor to [min, max] to prevent numerical instability.
        Inject random noise of magnitude noise to prevent tensor singularity, i.e. to avoid having
        linearly dependent row or column vectors in the Tensor.

        TODO: Identify optimal solutions for numerical instability for Gaussian EM.
        """
        # Identify the sign of the tensor.
        tsign = tensor.sign()
        # Compute the clamped absolute value of the tensor.
        tabs = tensor.abs().clamp(min=min, max=max)
        # Inject noise.
        perturb = torch.zeros(*tensor.shape)
        if noise is not None:
            # Warning: Noise <<< min in magnitude to insure numerical accuracy.
            perturb = 2 * noise * (torch.rand(*tensor.shape, generator=self.torch_gen) - 0.5)
            pMask = torch.zeros(*perturb.shape, dtype=torch.bool)
            # Identify zeros, clamped minimums, and clamped maximums.
            pMask = torch.logical_or(pMask, torch.eq(tsign, 0.0))
            if min is not None:
                pMask = torch.logical_or(pMask, torch.eq(tabs, min))
            if max is not None:
                pMask = torch.logical_or(pMask, torch.eq(tabs, max))
            perturb *= pMask
        # Return the magnitude-clamped noisy tensor.
        return tabs * tsign + perturb

class MatrixFactorize(Module):
    """
    Matrix Factorization Model for Numerical Factorization and Recommender Systems
    """

    def __init__(self, M: Union[torch.Tensor, np.ndarray], dim, mask: Union[torch.Tensor, np.ndarray] = None, bias=True, seed=None):
        """
        Instantiate variables for matrix factorization, i.e. M = A * B + C.
        :param M <torch.Tensor|numpy.ndarray>:      Input 2-D matrix to factorize into A and B (with bias C).
        :param dim <int>:                           Hidden / latent dimension of matrix factorization.
        :param mask <torch.Tensor|numpy.ndarray>:   Element observability mask matrix with shape equivalent to M. 
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
        mask: Union[torch.Tensor, np.ndarray] = None, 
        cycles=100, 
        lr=2e-3, 
        batch_frac=0.01, 
        regularize=1e-2, 
        patience=3, 
        verbose=False
    ):
        """
        Train the Factorization Machine.
        :param X <torch.Tensor>:        Input training data features of shape (N, X).
        :param Y <torch.Tensor>:        Target training data class / score vector of shape (N, 1).
        :param mask <torch.Tensor>:     Feature observability mask for X of shape (N, X).
        :param cycles <int>:            Number of gradient descent cycles.
        :param lr <float>:              Learning rate. Re-calibrated to order of values in matrix M.
        :param batch_frac <float>:      Fraction of the dataset to set as the batch size.
        :param regularize <float>:      Weight decay lambda for regularization in AdamW.
        :param patience <int>:          Number of cycles of convergence before termination.
        :param verbose <bool>:          Output training progress information.
        """

        # Validate arguments.
        if any([
            len(X.shape) != 2,
            len(Y.shape) != 2,
            mask is not None and mask.shape != X.shape,
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
                - X is a (N, data_dim) Tensor and Y is a (N, 1) Tensor.
                - mask is an optional (N, data_dim) Tensor.
                - cycles > 0
                - lr > 0
                - batch_frac > 0
                - regularize >= 0
                """,
                file=sys.stderr, flush=True
            )
            return

        # Convert and instantiate Tensor(s).
        N = X.shape[0]
        if not torch.is_tensor(X):
            X = torch.Tensor(X)
        if not torch.is_tensor(Y):
            Y = torch.Tensor(Y)
        mask_tensor = torch.ones(X.shape)
        if mask is not None:
            # Construct Boolean mask Tensor.
            mask_tensor = torch.where(torch.Tensor(mask) != 0, 1, 0)

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
                mask_batch = mask_tensor[rand_idx]

                # Clear gradients in optimizer.
                self.zero_grad()

                # Compute model prediction mean and deviation.
                Y_mu, Y_sigma = self(X_batch * mask_batch)

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