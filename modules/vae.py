import math
import torch
import torch.nn as nn

class VAE(nn.Module):
    """VAE with normal prior"""
    def __init__(self, encoder, decoder, args):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.adv = nn.Sequential(
                nn.Linear(args.nz, args.nz // 2),
                nn.Sigmoid(),
                nn.Linear(args.nz // 2, 1),
                nn.Sigmoid()
            )

        self.maeloss = nn.L1Loss(reduction="none")
        self.args = args
        self.nz = args.nz
        self.disen_dim = args.disen_dim
        self.att_dim = args.att_dim

        loc = torch.zeros(self.nz+self.disen_dim, device=args.device)
        scale = torch.ones(self.nz+self.disen_dim, device=args.device)
        att_loc = torch.zeros(self.att_dim, device=args.device)
        att_scale = torch.ones(self.att_dim, device=args.device)

        self.prior = torch.distributions.normal.Normal(loc, scale)
        self.att_prior = torch.distributions.normal.Normal(att_loc, att_scale)



    def encode_att(self, x, nsamples=1):
        """
        Returns: Tensor1, Tensor2
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
            Tensor2: the tenor of KL for each x with shape [batch]
        """
        return self.encoder.encode(x, nsamples), self.encoder.input_hidden() 


    def decode(self, z, y, strategy):
        """generate samples from z given strategy

        Args:
            z: [nsamples, nz]
            y: [nsamples, 1]
            strategy: "greedy" or "sample"
            K: the beam width parameter

        Returns: List1
            List1: a list of decoded word sequence
        """

        if strategy == "greedy":
                return self.decoder.greedy_decode(z, y)
        elif strategy == "sample":
            return self.decoder.sample_decode(z, y)
        else:
            raise ValueError("the decoding strategy is not supported")



    def loss_att_adv(self,x, y, nsamples=1):
        z, _ = self.encoder.encode(x, nsamples)
        batch_size, n_sample, _ = z.size()
        z = z.view(batch_size * n_sample, self.nz + self.disen_dim)
        adv_z = z[:,:-self.disen_dim]
        adv_predict = self.adv(adv_z.detach())
        adv_loss = self.maeloss(adv_predict, y)
        return adv_loss.squeeze()


    def loss_att(self, x, y, kl_weight, kl_att_weight, adv_weight, nsamples=1):
        (z, KL), all_hidden = self.encode_att(x, nsamples)
        reconstruct_err, att_KL, adv_h, dis_loss = self.decoder.reconstruct_error(x, z, all_hidden, y)
        adv_predict = self.adv(adv_h)
        adv_loss = self.maeloss(adv_predict, y)
        adv_loss = adv_loss.squeeze()
        dis_loss = dis_loss.squeeze()

        reconstruct_err = reconstruct_err.mean(dim=1)
        return reconstruct_err + kl_weight * (
                KL + kl_att_weight * att_KL) + adv_weight * (
                           dis_loss - adv_loss), reconstruct_err, KL, att_KL, adv_loss, dis_loss



    def sample_from_prior(self, nsamples):
        """sampling from prior distribution
        Returns: Tensor
            Tensor: samples from prior with shape (nsamples, nz)
        """
        return self.prior.sample((nsamples,))
