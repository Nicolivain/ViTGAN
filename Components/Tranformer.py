import torch
import torch.nn as nn

from Components.Attention import MultiHeadSelfAttentionL2
from Components.SLN import SLN
from Components.MLP import MLP


class Transformer(nn.Module):
    def __init__(self, in_features, n_head, attention_head_outdim=None, attention_dropout_rate=0.0, mlp_layers=None, mlp_activation='relu', mlp_dropout=0.0, spectral_rescaling=False):
        """
        Usual Transformer architecture using the L2-MultiheadSelfAttention module
        :param in_features: number of input features
        :param n_head: number of attention head
        :param attention_head_outdim: output size of each attention head, default is in_features // n_head
        :param attention_dropout_rate: dropout rate applied at the output of the MSA
        :param mlp_layers: list of hidden layer dimensions of the MLP module
        :param mlp_activation: activation function of the MLP module
        :param mlp_dropout: dropout applied at each MLP layer
        :param spectral_rescaling: use spectral rescaling in attention module
        """
        super(Transformer, self).__init__()

        self.in_features    = in_features
        self.n_head         = n_head
        self.head_outdim    = in_features // n_head if attention_head_outdim is None else attention_head_outdim

        self.ln1 = nn.LayerNorm(self.in_features)
        self.ln2 = nn.LayerNorm(self.in_features)

        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.msa = MultiHeadSelfAttentionL2(self.in_features, self.n_head, self.head_outdim, output_size=in_features, spectral_scaling=spectral_rescaling)
        self.mlp = MLP(self.in_features, self.in_features, layers=mlp_layers, activation=mlp_activation, dropout_rate=mlp_dropout)

    def forward(self, x):
        nx  = self.ln1(x)
        nx  = self.att_dropout(self.msa(nx))
        res = self.ln2(nx + x)
        out = self.mlp(res) + res
        return out


class TransformerSLN(nn.Module):
    def __init__(self, in_features, n_head, attention_head_outdim=None, attention_dropout_rate=0.0, mlp_layers=None, mlp_activation='relu', mlp_dropout=0.0, spectral_rescaling=False):
        """
        Variant Transformer architecture using the L2-MultiheadSelfAttention module and SLN instead of standard LayerNorm
        :param in_features: number of input features
        :param n_head: number of attention head
        :param attention_head_outdim: output size of each attention head, default is in_features // n_head
        :param attention_dropout_rate: dropout rate applied at the output of the MSA
        :param mlp_layers: list of hidden layer dimensions of the MLP module
        :param mlp_activation: activation function of the MLP module
        :param mlp_dropout: dropout applied at each MLP layer
        :param spectral_rescaling: use spectral rescaling in attention module
        """
        super(TransformerSLN, self).__init__()

        self.in_features    = in_features
        self.n_head         = n_head
        self.head_outdim    = in_features // n_head if attention_head_outdim is None else attention_head_outdim

        self.ln1 = SLN(self.in_features)
        self.ln2 = SLN(self.in_features)

        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.msa = MultiHeadSelfAttentionL2(self.in_features, self.n_head, self.head_outdim, output_size=in_features, spectral_scaling=spectral_rescaling)
        self.mlp = MLP(self.in_features, self.in_features, layers=mlp_layers, activation=mlp_activation, dropout_rate=mlp_dropout)

    def forward(self, h, x):
        nx  = self.ln1(h, x)
        nx  = self.att_dropout(self.msa(nx))
        res = self.ln2(h, nx + h)
        out = self.mlp(res) + res
        return out


if __name__ == '__main__':
    m = torch.randn(100, 10, 5)
    t = Transformer(5, 4, 3)
    ret = t(m)
    print(ret.shape)

    t = TransformerSLN(5, 4, 3)
    ret = t(m[0, :, :], m)
    print(ret.shape)
