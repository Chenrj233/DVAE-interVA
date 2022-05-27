from itertools import chain
import math
import torch.nn as nn

from .gaussian_encoder import GaussianEncoderBase

from sympy import *

class GaussianLSTMEncoder(GaussianEncoderBase):
    """Gaussian LSTM Encoder with constant-length input"""
    def __init__(self, args, vocab_size, model_init, emb_init):
        super(GaussianLSTMEncoder, self).__init__()
        self.ni = args.ni
        self.nh = args.enc_nh
        self.nz = args.nz
        self.disen_dim = args.disen_dim
        self.args = args

        self.embed = nn.Embedding(vocab_size, args.ni)

        self.lstm = nn.LSTM(input_size=args.ni,
                            hidden_size=args.enc_nh,
                            num_layers=1,
                            batch_first=True,
                            dropout=0)
        # dimension transformation to z (mean and logvar)
        self.linear = nn.Linear(args.enc_nh, 2 * (args.nz+args.disen_dim), bias=False)
        self.mu_bn = nn.BatchNorm1d(args.nz+args.disen_dim)
        self.mu_bn.weight.requires_grad = False

        self.reset_parameters(model_init, emb_init)

    def reset_parameters(self, model_init, emb_init, reset=False):
        if not reset:

            self.mu_bn.weight.fill_(self.args.gamma)
        else:
            print('reset bn!')
            self.mu_bn.weight.fill_(self.args.gamma)
            nn.init.constant_(self.mu_bn.bias, 0.0)

    def forward(self, input):
        """
        Args:
            x: (batch_size, seq_len)

        Returns: Tensor1, Tensor2
            Tensor1: the mean tensor, shape (batch, nz)
            Tensor2: the logvar tensor, shape (batch, nz)
        """

        word_embed = self.embed(input)

        _, (last_state, last_cell) = self.lstm(word_embed)
        self.all_hidden = _
        mean, logvar = self.linear(last_state).chunk(2, -1)
        if self.args.gamma > 0:
            mean = self.mu_bn(mean.squeeze(0))
        else:
            mean = mean.squeeze(0)
        # fix variance as a pre-defined value
        if self.args.fix_var > 0:
            logvar = mean.new_tensor([[[math.log(self.args.fix_var)]]]).expand_as(mean)
            
        return mean, logvar.squeeze(0)

    def input_hidden(self):
        return self.all_hidden



