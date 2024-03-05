"""
Machine Learning Library (Python v3.12.2+ / PyTorch 2.2.1+cu121)
Implemented by Cory Ye
For personal educational review.
"""
import sys
import math
import numpy as np
import torch
from AlgorithmLibrary import Numerics
from torch.autograd import Function, grad
from torch.nn import *
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from typing import Union, Callable, Tuple

class Transformer(Module):
    """
    Custom PyTorch implementation of Transformer. For more information on the model,
    refer to the paper "Attention Is All You Need": https://arxiv.org/pdf/1706.03762.pdf.
    """

    def __init__(self, encEmbed: torch.Tensor = None, decEmbed: torch.Tensor = None, maxLength: int = 1000, heads: int = 8, stack: int = 3, seed: int = None):
        super().__init__()
        # Fix random seed.
        if seed is not None:
            torch.manual_seed(seed)
        # Instantiate embedding spaces with consistent dimension.
        self.encEmbed: Embedding = Embedding.from_pretrained(encEmbed, freeze=True) if encEmbed is not None else Embedding(num_embeddings=100, embedding_dim=16)
        self.decEmbed: Embedding = Embedding.from_pretrained(decEmbed, freeze=True) if decEmbed is not None else Embedding(num_embeddings=100, embedding_dim=16)
        if self.decEmbed.embedding_dim != self.encEmbed.embedding_dim:
            raise ValueError(f"All embedding dimensions of the Transformer must be consistent.\n(Decoder Embed Dim: {self.decEmbed.embedding_dim}, Encoder Embed Dim: {self.encEmbed.embedding_dim})")
        # Instantiate components of Transformer.
        self.encoder: TransformerEncoderModule = TransformerEncoderModule(dim=self.encEmbed.embedding_dim, heads=heads, stack=stack)
        self.decoder: TransformerDecoderModule = TransformerDecoderModule(dim=self.decEmbed.embedding_dim, heads=heads, stack=stack)
        self.tokenLinear = Linear(maxLength * self.decEmbed.embedding_dim, self.decEmbed.num_embeddings)
        # Positional Encoding
        self.pe: PositionEncoder = PositionEncoder(maxLength, self.decEmbed.embedding_dim)

    def forward(self, prompt: torch.Tensor, response: torch.Tensor, promptMask: torch.Tensor = None, responseMask: torch.Tensor = None):
        # Encode. Memorize output of encoder in decoder.
        self.encode(prompt, promptMask)
        # Decode. Output generated token probabilities.
        return self.decode(response, responseMask)
    
    def encode(self, prompt: torch.Tensor, promptMask: torch.Tensor = None):
        # Lookup token embeddings.
        promptEmbed = self.encEmbed(prompt)
        # Apply positional encoding.
        promptPosEmbed = self.pe(promptEmbed)
        # Encode prompt.
        encodedPrompt = self.encoder(promptPosEmbed, promptMask)
        # Store encoded prompt into decoder.
        self.decoder.setMemory(encodedPrompt)
        # Return output of the encoder.
        return encodedPrompt

    def decode(self, response: torch.Tensor, responseMask: torch.Tensor = None):
        # Lookup token embeddings.
        responseEmbed = self.decEmbed(response)
        # Apply positional encoding.
        responsePosEmbed = self.pe(responseEmbed)
        # Decode the response. Utilizes memorized prompt encoding.
        decodedResponse = self.decoder(responsePosEmbed, responseMask)
        # Compute scores for all possible tokens. Because L is variable, infer from shape.
        # (N, L, D) -> Linear(L x D, T) -> (N, T)
        paddingShape = (0, self.tokenLinear.in_features - self.decEmbed.embedding_dim * decodedResponse.shape[-2])
        tokenScores = self.tokenLinear(
            functional.pad(decodedResponse.flatten(start_dim=-2), paddingShape, value=0.0)
        ).softmax(dim=-1)
        return tokenScores
    
    def train(
        self,
        trainData: Union[Tuple[torch.Tensor], torch.Tensor],
        evalData: Union[Tuple[torch.Tensor], torch.Tensor],
        numEpoch: int = 25,
        batchSize: int = 16,
        lr: float = 2e-3,
        seed: int = None
    ):
        # Setup training on GPU.
        device = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")

        # Build optimizer.
        adam = self.buildOptimizer(lr)

        # Setup loss function.
        loss = self.buildLossFunction()

        # Construct Datasets.
        tpData, trData, epData, erData = None, None, None, None
        if isinstance(trainData, tuple) and len(trainData) == 2:
            tpData = TokenSequence(trainData[0], maskedSequence=False)
            trData = TokenSequence(trainData[1], maskedSequence=True)
        else:
            tpData, trData = TokenSequence.dualTokenSequence(trainData, shuffle=True, seed=seed)
        if isinstance(evalData, tuple) and len(evalData) == 2:
            epData = TokenSequence(evalData[0], maskedSequence=False)
            erData = TokenSequence(evalData[0], maskedSequence=True)
        else:
            epData, erData = TokenSequence.dualTokenSequence(evalData, shuffle=True, seed=seed+1)

        # Construct DataLoader.
        tpLoader = DataLoader(dataset=tpData, batch_size=batchSize, pin_memory=True, num_workers=4*torch.cuda.device_count())
        trLoader = DataLoader(dataset=trData, batch_size=batchSize, pin_memory=True, num_workers=4*torch.cuda.device_count())

        # Train.
        for epoch in range(numEpoch):
            for i, batch in enumerate(zip(tpLoader, trLoader)):

                # Unpack prompt and response batch.
                pmptBatch, respBatch = batch

                # Unpack input sequence, generated labels, and mask for response.
                resp, gen, rMask = respBatch

                # Unpack input sequence and mask for prompt.
                pmpt, _, pMask = pmptBatch

                # Zero gradients of model.
                self.zero_grad()

                # Forward propagate.
                outputTokenProb = self(pmpt, resp, pMask, rMask)

                # Compute loss.
                batchLoss = loss(outputTokenProb, gen).sum()
                print(f"Epoch: {epoch+1} / {numEpoch} | Batch: {i} / {len(tpLoader)} | Loss: {batchLoss}")

                # Compute gradients.
                batchLoss.backward()

                # Apply gradients to optimize loss.
                adam.step()
        
    def buildLossFunction(self):
        return CrossEntropyLoss()

    def buildOptimizer(self, lr: float, beta1: float = 0.9, beta2: float = 0.98):
        return AdamW(self.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=1e-2)

class TransformerEncoderModule(Module):
    """
    Custom PyTorch implementation of TransformerEncoder.
    """
    def __init__(self, dim: int, heads: int, stack: int = 1):
        super().__init__()
        # Instantiate parameters for self-attention and dense residual sub-blocks.
        self.multiHead = ModuleList([MultiHeadAttention(dim, dim, dim, heads) for _ in range(stack)])
        self.atxResidual = ModuleList([
            ResidualNorm(lambda x, mask: self.multiHead[i](x, x, x, mask))
            for i in range(stack)
        ])
        self.linearStack = ModuleList([Linear(dim, dim) for _ in range(2*stack)])
        self.denseResidual = ModuleList([
            ResidualNorm(lambda x, _: self.linearStack[2*i](LeakyReLU()(self.linearStack[2*i+1](x))))
            for i in range(stack)
        ])

    def forward(self, prompt: torch.Tensor, mask: torch.Tensor = None):
        # Encode the prompt.
        x = prompt
        # Loop through series of encoder blocks.
        for atx, dense in zip(self.atxResidual, self.denseResidual):
            # Self-Attention
            atxOutput = atx(x, mask)
            # Dense
            x = dense(atxOutput, None)
        # Return encoded output embeddings.
        return x

class TransformerDecoderModule(Module):
    """
    Custom PyTorch implementation of TransformerDecoder.
    """
    def __init__(self, dim: int, heads: int, encoderMemory: torch.Tensor = None, stack: int = 1):
        super().__init__()
        # Store encoder output.
        self.encMem = encoderMemory
        # Instantiate parameters for self-attention, encoder-decoder attention, and dense residual sub-blocks.
        self.multiHead = ModuleList([MultiHeadAttention(dim, dim, dim, heads) for _ in range(2*stack)])
        self.selfAtxResidual = ModuleList([
            ResidualNorm(lambda x, mask: self.multiHead[2*i](x, x, x, mask))
            for i in range(stack)
        ])
        self.encAtxResidual = ModuleList([
            # Note: Encoding tensor only needs to be computed once and memorized.
            ResidualNorm(lambda x, _: self.multiHead[2*i+1](
                # Default to self-attention if encoder output is not provided.
                x, self.encMem if self.encMem is not None else x, self.encMem if self.encMem is not None else x
            ))
            for i in range(stack)
        ])
        self.linearStack = ModuleList([Linear(dim, dim) for _ in range(2*stack)])
        self.denseResidual = ModuleList([
            ResidualNorm(lambda x, _: self.linearStack[2*i](LeakyReLU()(self.linearStack[2*i+1](x))))
            for i in range(stack)
        ])

    def forward(self, response: torch.Tensor, mask: torch.Tensor = None):
        # Given the (previous) response, compute an embedding that attends to the prompt.
        x = response
        for selfAtx, encAtx, dense in zip(self.selfAtxResidual, self.encAtxResidual, self.denseResidual):
            # Self-Attention
            selfAtxOutput = selfAtx(x, mask)
            # Encoder-Decoder Attention
            encAtxOutput = encAtx(selfAtxOutput, None)
            # Dense
            x = dense(encAtxOutput, None)
        # Return decoded output embeddings.
        return x
    
    def setMemory(self, encoderMemory: torch.Tensor):
        self.encMem = encoderMemory
    
class MultiHeadAttention(Module):
    """
    Multi-Head Attention Mechanism
    """
    def __init__(self, dq: int, dk: int, dv: int, heads: int):
        super().__init__()
        # Module parameters.
        self.numHeads = heads
        self.qkvcLinear = ModuleList()
        self.headDims = []
        # Instantiate heads for QKV.
        for dim in [dq, dk, dv]:
            # Validate head dimension divisibility.
            if dim % heads != 0 or dim // heads == 0:
                raise ValueError(f"Number of heads ({heads}) should divide the dimension ({dim}) of QKV.")
            # Instantiate head projections. Because the head projection dimensions
            # add up to dModel, we can simultaneously compute all head projections
            # and reshape the output to dHead to retrieve them.
            self.headDims.append(dim // heads)
            self.qkvcLinear.append(Linear(dim, dim))
        # Instantiate head for dense aggrevation layer.
        self.qkvcLinear.append(Linear(dv, dv))

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
        # Deduce the number of samples in the batch N.
        N = q.shape[0]
        # Apply projections to Q, K, and V.
        # (N, L, D) -> (N, H, L, D // H)
        query, key, value = [
            self.qkvcLinear[i](x).view(N, -1, self.numHeads, self.headDims[i]).transpose(1,2)
            for i, x in enumerate([q, k, v])
        ]
        # Batch-apply attention to Q, K, and V.
        # (N, H, L, D // H) -> (N, H, LQ, DV // H)
        headMask = mask.view(N, 1, q.shape[1], k.shape[1]) if mask is not None else None
        atxOutput = Attention.apply(query, key, value, headMask)
        # Concatenate.
        # (N, H, LQ, DV // H) -> (N, LQ, DV)
        atxConcat = atxOutput.transpose(1,2).flatten(start_dim=2)
        # Compute dense aggregation layer.
        # (N, LQ, DV) -> (N, LQ, DV)
        return self.qkvcLinear[-1](atxConcat)
    
class Attention(Function):
    """
    PyTorch implementation of a Q-K-V Scaled Dot-Product Attention Block.
    """
    @staticmethod
    @torch.no_grad
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
        # Validate DQ == DK and LK == LV.
        if q.shape[-1] != k.shape[-1] or k.shape[-2] != v.shape[-2]:
            # Inconsistent embedding or sequence dimensions for attention mechanism.
            raise ValueError(f"[ATXDimError] Require DQ ({q.shape[-1]}) == DK ({k.shape[-1]}) and LK ({k.shape[-2]}) == LV ({v.shape[-2]}) with input dimensions (N,L,D).")
        DK = k.shape[-1]
        # Compute co-similarity matrix QK.
        # (N, *, LQ, LK) = (N, *, LQ, DQ) x (N, *, DK, LK)
        qk = torch.matmul(q, k.transpose(-1, -2))
        # Scale co-similarity coefficients by sqrt(DK).
        qkScaled = qk / torch.sqrt(torch.tensor([DK]))
        # Apply mask, i.e. reduce the co-similarity score to a large negative number,
        # which annihilates when the exponential function is applied during softmax.
        if mask is not None:
            # Braodcastable masking function, i.e. mask across (N, *, LQ, LK).
            qkScaled = qkScaled.masked_fill(mask.logical_not(), float("-inf"))
        # Apply softmax across the dimension of K to normalize the convex combination of
        # value embeddings V weighted by the similarity probability of key embeddings K
        # for each query embedding in Q, i.e. V_i scaled by the probability or proportion
        # of which K_i composes Q in order to have embeddings in the space of V that
        # represent or are attentive to the embeddings in the space of Q.
        qkSoftmax = qkScaled.softmax(dim=-1).nan_to_num(nan=0.0)
        # Compute the QK-convex combination of value embeddings V.
        # (N, *, LQ, DV) = (N, *, LQ, LK) x (N, *, LV, DV)
        output = torch.matmul(qkSoftmax, v)
        # Save tensors for backwards pass.
        ctx.save_for_backward(q, k, v, qkSoftmax, mask)
        # Return output embeddings and attention weights.
        return output
    
    @staticmethod
    @torch.no_grad
    def backward(ctx, doutput: torch.Tensor):
        # Unpack the forward propagation values.
        q, k, v, qkSoftmax, mask = ctx.saved_tensors
        N = qkSoftmax.shape[0]
        LQ = qkSoftmax.shape[-2]
        LK = qkSoftmax.shape[-1]
        DK = k.shape[-1]
        # Backpropagate torch.matmul(qkSoftmax, v).
        dv = torch.matmul(qkSoftmax.transpose(-1,-2), doutput)      # (N, *, LV, DV) = (N, *, LK, LQ) x (N, *, LQ, DV)
        dqkSoftmax = torch.matmul(doutput, v.transpose(-1,-2))      # (N, *, LQ, LK) = (N, *, LQ, DV) x (N, *, DV, LV)
        # Per column, the Jacobian partial derivatives for the Softmax,
        # where S represents "scores" and P represents "softmax probabilities":
        # dpi/dsi = e^{si} * sum_{!i}(e^{sj}) / sum(e^{sj})^2 = pi * (1 - pi)
        # dpi/dsj = - e^{si} * e^{sj} / sum(e^{sj})^2 = - pi * pj
        # which simplifies to: Diag(P) - P x P^{T} for each dimension of Q.
        diagSoftmax = qkSoftmax.view(N,-1,LQ,LK,1) * torch.diag(torch.ones(LK))
        dSoftmax = diagSoftmax - torch.matmul(qkSoftmax.view(N,-1,LQ,LK,1), qkSoftmax.view(N,-1,LQ,1,LK))
        # Backpropagate qkScaled.softmax(dim=-1).
        # dqkScaled = dSoftmax(qkScaled) x dqkSoftmax
        # (N, *, LQ, LK) = (N, *, LQ, LK, LK) x (N, *, LQ, LK, 1)
        dqkScaled = torch.matmul(dSoftmax, dqkSoftmax.view(N,-1,LQ,LK,1)).view(N,-1,LQ,LK)
        # Backpropagate the constant mask.
        if mask is not None:
            # Derivative of a (large negative) constant is 0.
            dqkScaled = dqkScaled.masked_fill(mask.logical_not(), 0.0)
        # Backpropagate the scaling.
        dqk = dqkScaled / torch.sqrt(torch.tensor([DK]))
        # Backpropagate the attention co-similarity matrix.
        dq = torch.matmul(dqk, k)                       # (N, *, LQ, DQ) = (N, *, LQ, LK) x (N, *, LK, DK)
        dk = torch.matmul(dqk.transpose(-1,-2), q)      # (N, *, LK, DK) = (N, *, LK, LQ) x (N, *, LQ, DQ)
        # Return derivatives.
        return dq, dk, dv, None
    
class ResidualNorm(Module):
    """
    Attached Residual + Normalization Layer for Transformer.

    For more information on LayerNorm, refer to:
    https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    """
    def __init__(self, baseModule: Union[Module, Callable]):
        super().__init__()
        self.baseModule = baseModule
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Compute LayerNorm(x + baseModule(x)).
        """
        return LayerNorm(x.shape[-1])(x + self.baseModule(x, mask))
    
class PositionEncoder(Module):
    """
    Add a positional encoding based on 2-D rotations (SO(2)) to the input Tensor.
    Assumes that x has the shape (*, L, D) where L represents the positional dimension
    and D represents the embedding dimension.

    For each pair of indices (2i,2i+1) in the embedding dimension d at position k,
    and with trigonometric "base" period t, we have:

    P(k,2i)   = sin(k/t^{2i/d})
    P(k,2i+1) = cos(k/t^{2i/d})

    When the positional encoding is combined with the input embedding vectors,
    subsequent inner products between position-encoded embedding vectors will
    invoke the following angle difference formula, e.g. if d = 2 we have:
    
    <x(k),y(l)> = <(sin(k), cos(k)), (sin(l), cos(l))>
                = sin(k) sin(l) + cos(k) cos(l)
                = cos(k - l) = cos(l - k)

    Model weights sensitive to these inner products from self-attention layers
    can learn positional features through the trigonometric encoding cos(|k-l|).

    Note that because the period of the encoding is extremely large, i.e. on the
    order of t^{O(L)} where t is typically chosen to be sufficiently large, this
    encoding supports a large quantity of unique positional encoding values before
    the encoding "overflows" into the next period of the trigonometric sinusoids,
    in which case such tokens close and far will be indistinguishable.
    """

    def __init__(self, maxLength: int, maxDim: int, t: int = 10000):
        super().__init__()
        self.maxLength = maxLength
        self.maxDim = maxDim
        # Positions to encode.
        # Shape: (maxLength, 1)
        posArray = torch.arange(0, maxLength).view(maxLength, 1)
        # Dimension indices to encode.
        # Shape: (1, maxDim)
        halfDim = maxDim/2
        dimArray = torch.pow(torch.tensor([t]), 2 * torch.arange(0, math.ceil(halfDim)) / maxDim).view(1, -1)
        # Cache the positional encoding tensor.
        # Shape: (maxLength, maxDim)
        self.positionalEncoding = torch.zeros(maxLength, maxDim)
        self.positionalEncoding[:,0::2] = torch.sin(torch.div(posArray, dimArray[:,:math.ceil(halfDim)]))
        self.positionalEncoding[:,1::2] = torch.cos(torch.div(posArray, dimArray[:,:math.floor(halfDim)]))

    def forward(self, x: torch.Tensor):
        """
        Compute x + self.positionalEncoding. Truncate to match the shape of x.
        """
        return x + self.positionalEncoding[:x.shape[-2],:x.shape[-1]]
    
    def getMaxLength(self):
        return self.maxLength
    
    def getMaxDim(self):
        return self.maxDim

class TokenSequence(Dataset):

    def __init__(self, data: torch.Tensor, maskedSequence: bool = False):
        # Store token sequences.
        # Shape: (N, L)
        self.seqs = data
        # Store masked response sequences if maskedSequence,
        # else store duplicated prompt sequences x (L-1).
        self.X = self._triangleExpand(data) if maskedSequence else self._duplExpand(data)
        # Store generated token labels.
        self.Y = self._labelExpand(data)
        # Compute batched masks for X.
        seqMask = self.X.unsqueeze(-1)
        # (N, L, L) = (N, L, 1) x (N, 1, L)
        self.triMask = torch.matmul(seqMask, seqMask.transpose(-1,-2)).bool()

    @staticmethod
    def dualTokenSequence(data: torch.Tensor, shuffle: bool = False, seed: int = None):
        """
        Static Dual-Prompt/Response Dataset Builder for TokenSequence
        """
        # Shuffle and partition dataset of prompts and responses.
        # Shape: (N, 2, L)
        dual = data
        if shuffle:
            randomShuffle = Numerics.randomSample(data.shape[0], seed=seed)
            dual = data[randomShuffle]
        # Build Prompt and Response TokenSequence Datasets.
        return (
            # Prompt
            TokenSequence(dual[:,0,:], maskedSequence=False),
            # Response
            TokenSequence(dual[:,1,:], maskedSequence=True),
        )

    def __len__(self):
        # Length of expanded masked sequence data.
        return self.X.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Retrieve input, output, and regressive mask.
        return self.X[idx], self.Y[idx], self.triMask[idx]
    
    def _triangleExpand(self, data: torch.Tensor):
        """
        Expand sequences into previous tokens.
        """
        # Shape: [N x (L-1), L]
        return data.unsqueeze(-2).expand(-1,self.seqs.shape[-1],-1).tril()[:,:-1,:].flatten(0,1)
    
    def _labelExpand(self, data: torch.Tensor):
        """
        Expand sequences into a generated label.
        """
        # Shape: [N x (L-1)]
        return data[:,1:].flatten(0,1)
    
    def _duplExpand(self, data: torch.Tensor):
        """
        Duplicate sequences x (L-1).
        """
        # Shape: [N x (L-1)]
        return data.unsqueeze(-2).expand(-1,self.seqs.shape[-1]-1,-1).flatten(0,1)

    def shape(self):
        return self.seqs.shape
    
    def getSize(self):
        return self.shape()[0]
    
    def getData(self):
        return self.seqs
    
    def getX(self):
        return self.X
    
    def getY(self):
        return self.Y
    
    def getMask(self):
        return self.triMask

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
        self.data = torch.tensor(data)
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
            data_reshaped = self.data.view(1, self.N, self.D, 1)
            mu_reshaped = self.mu.view(self.K, 1, self.D, 1)
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
        x_reshaped = x.view(M, 1, self.D, 1)
        mu_reshaped = mu.view(1, self.K, self.D, 1)
        sigma_reshaped = sigma.view(1, self.K, self.D, self.D)
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
        self.M = torch.tensor(M)
        self.x, self.y = self.M.shape
        self.dim = dim
        self.bias = bias
        self.mask = torch.ones(self.M.shape)
        if mask is not None and mask.shape == self.M.shape:
            # Construct Boolean mask Tensor.
            self.mask = torch.where(torch.tensor(mask) != 0, 1, 0)

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
            X = torch.tensor(X)
        if not torch.is_tensor(Y):
            Y = torch.tensor(Y)
        mask_tensor = torch.ones(X.shape)
        if mask is not None:
            # Construct Boolean mask Tensor.
            mask_tensor = torch.where(torch.tensor(mask) != 0, 1, 0)

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
