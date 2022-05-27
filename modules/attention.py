import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class LuongAttention(nn.Module):
    """
    Note here that we are only implementing the 'general' method as denoted in the paper
    """

    def __init__(self, hidden_size_enc, hidden_size_dec, use_cuda=True, method='general'):
        super(LuongAttention, self).__init__()
        self.hidden_size_enc = hidden_size_enc
        self.hidden_size_dec = hidden_size_dec
        self.use_cuda = use_cuda
        self.method = method
        if self.method not in ['general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        if self.method == 'general':
            self.general_weights = torch.nn.Parameter(torch.randn(hidden_size_dec, hidden_size_enc))
        elif self.method == 'concat':
            self.general_weights = Variable(torch.randn(hidden_size_dec, hidden_size_enc))
            self.v = Variable(torch.randn(hidden_size_dec, hidden_size_enc))

    def forward(self,
                encoder_outputs,
                decoder_outputs,
                enc_mask=None):

        dec_len = decoder_outputs.size(1)
        enc_len = encoder_outputs.size(1)

        encoder_outputs = encoder_outputs.permute(0, 2, 1)
        score = torch.bmm(decoder_outputs @ self.general_weights, encoder_outputs)

        if enc_mask is not None:
            enc_mask = enc_mask.unsqueeze(1)
            enc_mask = torch.transpose(enc_mask, 0, 2)
            score = score.masked_fill(enc_mask == 0, -1e12)


        weights_flat = F.softmax(score.view(-1, enc_len), dim=1)
        weights = weights_flat.view(-1, dec_len, enc_len)
        attention_vector = torch.bmm(weights, encoder_outputs.permute(0, 2, 1))
        attention_vector = attention_vector.permute(1, 0, 2)

        return attention_vector, weights.view(-1, enc_len)
