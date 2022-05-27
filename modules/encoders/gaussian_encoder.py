import math
import torch


from .encoder import EncoderBase
from ..utils import log_sum_exp

class GaussianEncoderBase(EncoderBase):
    """docstring for EncoderBase"""
    def __init__(self):
        super(GaussianEncoderBase, self).__init__()

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Args:
            x: (batch_size, *)

        Returns: Tensor1, Tensor2
            Tensor1: the mean tensor, shape (batch, nz)
            Tensor2: the logvar tensor, shape (batch, nz)
        """

        raise NotImplementedError

    def encode_stats(self, x):

        return self.forward(x)

    def sample(self, input, nsamples):
        """sampling from the encoder
        Returns: Tensor1
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
        """

        mu, logvar = self.forward(input)

        z = self.reparameterize(mu, logvar, nsamples)

        return z, (mu, logvar)

    def encode(self, input, nsamples):
        """perform the encoding and compute the KL term

        Returns: Tensor1, Tensor2
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
            Tensor2: the tenor of KL for each x with shape [batch]

        """

        mu, logvar = self.forward(input)

        z = self.reparameterize(mu, logvar, nsamples)

        KL = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(dim=1)


        return z, KL

    def reparameterize(self, mu, logvar, nsamples=1):
        """sample from posterior Gaussian family
        Args:
            mu: Tensor
                Mean of gaussian distribution with shape (batch, nz)

            logvar: Tensor
                logvar of gaussian distibution with shape (batch, nz)

        Returns: Tensor
            Sampled z with shape (batch, nsamples, nz)
        """
        batch_size, nz = mu.size()
        std = logvar.mul(0.5).exp()

        mu_expd = mu.unsqueeze(1).expand(batch_size, nsamples, nz)
        std_expd = std.unsqueeze(1).expand(batch_size, nsamples, nz)

        eps = torch.zeros_like(std_expd).normal_()

        return mu_expd + torch.mul(eps, std_expd)


