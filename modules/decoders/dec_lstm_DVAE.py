# import torch

import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..attention import *
import math

from .decoder import DecoderBase


class LSTMOurDecoder(DecoderBase):
    """LSTM decoder with constant-length data and attention"""
    def __init__(self, args, vocab, model_init, emb_init):
        super(LSTMOurDecoder, self).__init__()
        self.ni = args.ni
        self.nh = args.dec_nh
        self.nz = args.nz
        self.vocab = vocab
        self.device = args.device
        self.att_dim = args.att_dim
        self.sentiemb_dim = args.sentiemb_dim
        self.disen_dim = args.disen_dim
        self.args = args

        # no padding when setting padding_idx to -1
        self.embed = nn.Embedding(len(vocab), args.ni, padding_idx=-1)
        self.senti_embed = nn.Embedding(11, args.sentiemb_dim)

        self.dropout_in = nn.Dropout(args.dec_dropout_in)
        self.dropout_out = nn.Dropout(args.dec_dropout_out)

        # for initializing hidden state and cell  encoder content z -> decoder init_h
        self.trans_linear = nn.Linear(args.nz, args.dec_nh, bias=False)

        self.attn_linear = nn.Linear(args.dec_nh, 2 * args.att_dim, bias=False)
        self.attn = LuongAttention(args.enc_nh, args.dec_nh)

        self.disen_to_inten = nn.Sequential(
                nn.Linear(args.disen_dim, args.disen_dim // 2),
                nn.Sigmoid(),
                nn.Linear(args.disen_dim // 2, 1),
                nn.Sigmoid()
            )
        self.maeloss = nn.L1Loss(reduction="none")

        self.lstm = nn.LSTMCell(input_size=args.ni + args.att_dim + args.sentiemb_dim,
                            hidden_size=args.dec_nh)
        self.update_senti = nn.LSTMCell(input_size=args.dec_nh,
                                    hidden_size=args.sentiemb_dim)

        self.mu_bn = nn.BatchNorm1d(args.att_dim)
        self.mu_bn.weight.requires_grad = False

        self.out_weight = nn.Linear(args.sentiemb_dim, args.dec_nh, bias=False)
        self.pred_linear = nn.Linear(args.dec_nh, len(vocab), bias=False)

        vocab_mask = torch.ones(len(vocab))
        self.loss = nn.CrossEntropyLoss(weight=vocab_mask, reduce=False)

        self.reset_parameters(model_init, emb_init)

        att_loc = torch.zeros(self.att_dim, device=args.device)
        att_scale = torch.ones(self.att_dim, device=args.device)


        self.att_prior = torch.distributions.normal.Normal(att_loc, att_scale)

    def reset_parameters(self, model_init, reset=False):

        for param in self.parameters():
            model_init(param)
        nn.init.xavier_uniform_(self.embed.weight)
        nn.init.xavier_uniform_(self.senti_embed.weight)

        if not reset:
            self.mu_bn.weight.fill_(self.args.att_gamma)
        else:
            print('reset bn!')
            self.mu_bn.weight.fill_(self.args.att_gamma)
            nn.init.constant_(self.mu_bn.bias, 0.0)


    def generate_sentiment_intensity(self, disentangle):
        intensity = self.disen_to_inten(disentangle)
        return intensity

    def generate_sentiment_embedding(self, intensity):
        tmp_intensity = 10.0 * intensity
        trunc_intensity = torch.trunc(tmp_intensity).type(torch.long)
        frac_intensity = torch.frac(tmp_intensity)
        sen_emb = (-frac_intensity+1.0) * self.senti_embed(trunc_intensity).squeeze(1) + \
                         frac_intensity * self.senti_embed(torch.min(trunc_intensity + 1, 10*torch.ones(intensity.size(0), 1, device=self.device).type(torch.int))).squeeze(1)

        return sen_emb, intensity

    def get_sentiment_embbeding(self, disentangle):
        """
        Args:
            disentangle: (batch_size, disentangle_dim)
        """
        return self.generate_sentiment_embedding(self.generate_sentiment_intensity(disentangle))

    def sample_att_from_prior(self,nsamples=1):
        return self.att_prior.sample((nsamples,))

    def decode(self, input, z, all_hidden, y):

        batch_size, n_sample, _ = z.size()

        word_embed = self.embed(input)
        word_embed = self.dropout_in(word_embed)

        z = z.view(batch_size * n_sample, self.nz + self.disen_dim)
        dis_z = z[:, -self.disen_dim:]
        con_z = z[:, :-self.disen_dim]
        adv_h = con_z
        c_init = self.trans_linear(con_z)
        h_init = torch.tanh(c_init)

        senti_embed_dis, intensity_predict = self.get_sentiment_embbeding(dis_z)
        senti_embed_true, _ = self.generate_sentiment_embedding(y)
        ori_embed_true = senti_embed_true
        dis_loss = self.maeloss(intensity_predict, y)
        all_output = []


        for i in range(word_embed.size(1)):
            attn_vec, weights = self.attn(all_hidden, h_init.unsqueeze(1))
            attn_mu, attn_logvar = self.attn_linear(attn_vec).chunk(2, -1)
            attn_mu = attn_mu.squeeze(0)
            attn_mu = self.mu_bn(attn_mu)
            attn_logvar = attn_logvar.squeeze(0)
            # fix variance as a pre-defined value
            if self.args.fix_att_var > 0:
                attn_logvar = attn_mu.new_tensor([[[math.log(self.args.fix_att_var)]]]).expand_as(attn_mu)


            sample_attn_vec = self.reparameterize(attn_mu, attn_logvar)
            sample_attn_vec = sample_attn_vec.squeeze(1)
            attn_to_h = sample_attn_vec
            if i == 0:
                select = torch.rand(senti_embed_true.size(0),senti_embed_true.size(1)).cuda()
                senti_embed = torch.where(select > 0.5, senti_embed_true, senti_embed_dis)
                h_sen = torch.tanh(senti_embed)
                input_word_embed = torch.cat((word_embed[:, i, :], attn_to_h, senti_embed), -1)

            else:
                input_word_embed = torch.cat((word_embed[:, i, :], attn_to_h, senti_embed), -1)

            h_init, c_init = self.lstm(input_word_embed, (h_init, c_init))

            h_sen, senti_embed = self.update_senti(h_init, (h_sen, senti_embed))

            if i == 0:
                att_KL = 0.5 * torch.sum(torch.exp(attn_logvar) + attn_mu ** 2 - 1 - attn_logvar, 1)
            else:
                att_KL = att_KL + 0.5 * torch.sum(torch.exp(attn_logvar) + attn_mu ** 2 - 1 - attn_logvar, 1)

            all_output.append(h_init)

        cat_output = torch.stack(all_output, dim=1)
        senti = ori_embed_true.unsqueeze(1).repeat(1, cat_output.size(1), 1)
        weight_h = self.out_weight(senti)
        final_h = weight_h * cat_output

        output = self.dropout_out(final_h)
        output_logits = self.pred_linear(output)

        return output_logits, att_KL, adv_h, dis_loss

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


    def reconstruct_error(self, x, z, all_hidden, y):
        """Cross Entropy in the language case
        Args:
            x: (batch_size, seq_len)
            z: (batch_size, n_sample, nz)
            all_hidden: (batch_size, seq_len, hidden_state)
        Returns:
            loss: (batch_size, n_sample). Loss
            across different sentence and z
        """

        #remove end symbol
        src = x[:, :-1]

        # remove start symbol
        tgt = x[:, 1:]

        batch_size, seq_len = src.size()
        n_sample = z.size(1)
        output_logits, att_KL, adv_h, dis_loss = self.decode(src, z, all_hidden, y)


        if n_sample == 1:
            tgt = tgt.contiguous().view(-1)
        else:
            tgt = tgt.unsqueeze(1).expand(batch_size, n_sample, seq_len) \
                     .contiguous().view(-1)
        loss = self.loss(output_logits.view(-1, output_logits.size(2)),
                         tgt)


        return loss.view(batch_size, n_sample, -1).sum(-1), att_KL, adv_h, dis_loss

    def log_probability(self, x, z):
        """Cross Entropy in the language case
        Args:
            x: (batch_size, seq_len)
            z: (batch_size, n_sample, nz)
        Returns:
            log_p: (batch_size, n_sample).
                log_p(x|z) across different x and z
        """

        return -self.reconstruct_error(x, z)


    def greedy_decode(self, z, y):
        return self.sample_decode(z, y, greedy=True)

    def sample_decode(self, z, y, greedy=False):
        """sample/greedy decoding from z
        Args:
            z: (batch_size, nz)
            y: (batch_size, 1)

        Returns: List1
            List1: the decoded word sentence list
        """

        batch_size = z.size(0)
        decoded_batch = [[] for _ in range(batch_size)]

        c_init = self.trans_linear(z[:, :-self.disen_dim])
        h_init = torch.tanh(c_init)
        senti_embed_true, _ = self.generate_sentiment_embedding(y)
        ori_emb_true = senti_embed_true

        h_sen = torch.tanh(senti_embed_true)

        decoder_hidden = (h_init, c_init)
        decoder_input = torch.tensor([self.vocab["<s>"]] * batch_size, dtype=torch.long, device=self.device).unsqueeze(1)
        end_symbol = torch.tensor([self.vocab["</s>"]] * batch_size, dtype=torch.long, device=self.device)

        mask = torch.ones((batch_size), dtype=torch.uint8, device=self.device)
        length_c = 1

        while mask.sum().item() != 0 and length_c < 100:
            word_embed = self.embed(decoder_input)
            word_embed = word_embed.squeeze(1)
            sample_attn_vec = self.sample_att_from_prior(batch_size)

            word_embed = torch.cat((word_embed, sample_attn_vec, senti_embed_true), dim=-1)

            decoder_hidden = self.lstm(word_embed, decoder_hidden)
            output = decoder_hidden[0]
            weight_h = self.out_weight(ori_emb_true)
            h_sen, senti_embed_true = self.update_senti(output, (h_sen, senti_embed_true))
            final_h = weight_h * output
            decoder_output = self.pred_linear(final_h)
            output_logits = decoder_output


            if greedy:
                max_index = torch.argmax(output_logits, dim=1)
            else:
                probs = F.softmax(output_logits, dim=1)
                max_index = torch.multinomial(probs, num_samples=1).squeeze(1)

            decoder_input = max_index.unsqueeze(1)
            length_c += 1

            for i in range(batch_size):
                if mask[i].item():
                    decoded_batch[i].append(self.vocab.id2word(max_index[i].item()))

            mask = torch.mul((max_index != end_symbol), mask)

            return decoded_batch


