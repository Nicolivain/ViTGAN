import torch
import torch.nn as nn
from einops import rearrange, repeat


class AttentionL2(nn.Module):
    def __init__(self, in_features, out_features, scale=None, spectral_scaling=False, lp=2, **kwargs):
        """
        Single head L2-attention module
        :param in_features: number of input features
        :param out_features: number of out features
        :param scale: rescale factor of d(K,Q), default is out_features
        :param spectral_scaling: perform spectral rescaling of q, k, v linear layers at each forward call
        :param lp: norm used for attention, should be 1 or 2, default 2
        """
        super(AttentionL2, self).__init__()

        self.in_features      = in_features
        self.out_features     = out_features
        self.scale            = out_features if scale is None else scale
        self.spectral_scaling = spectral_scaling
        self.lp               = lp

        self.q = nn.Linear(self.in_features, self.out_features, bias=False)
        self.k = nn.Linear(self.in_features, self.out_features, bias=False)
        self.v = nn.Linear(self.in_features, self.out_features, bias=False)

        if self.spectral_scaling:
            sq, sk, sv = self._get_spectrum()
            self.init_spectrum = [max(sq), max(sk), max(sv)]

        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        if self.spectral_scaling:
            self._weight_spectral_rescale()
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        l2_dist = torch.cdist(q, k, p=self.lp)  # we use L2 reg only in discriminator
        att = self.softmax(l2_dist / (self.scale**(1/2))) @ v
        return att

    def _get_spectrum(self):
        _, sq, _ = torch.svd(self.q.weight)
        _, sk, _ = torch.svd(self.k.weight)
        _, sv, _ = torch.svd(self.v.weight)
        return sq, sk, sv

    def _weight_spectral_rescale(self):
        sq, sk, sv = self._get_spectrum()
        self.q.weight = nn.Parameter(max(sq) / self.init_spectrum[0] * self.q.weight)
        self.k.weight = nn.Parameter(max(sk) / self.init_spectrum[1] * self.k.weight)
        self.v.weight = nn.Parameter(max(sv) / self.init_spectrum[2] * self.v.weight)


class MultiHeadSelfAttentionL2(nn.Module):
    def __init__(self, in_features, n_head, head_dim, output_size=None, spectral_scaling=False, lp=2, **kwargs):
        """
        Multihead self L2-attention module based on L2-attention module
        :param in_features: number of input features
        :param n_head: number of attention heads
        :param head_dim: output size of each attention head
        :param output_size: final output feature number, default is n_head * head_dim
        :param spectral_scaling: perform spectral rescaling of q, k, v for each head
        :param lp: norm used for attention, should be 1 or 2, default 2
        """
        super(MultiHeadSelfAttentionL2, self).__init__()

        self.out_dim         = n_head * head_dim
        self.out_features    = self.out_dim if output_size is None else output_size

        self.attention_heads = nn.ModuleList([AttentionL2(in_features, head_dim, scale=self.out_dim, spectral_scaling=spectral_scaling, lp=lp) for _ in range(n_head)])
        self.output_linear   = nn.Linear(self.out_dim, self.out_features)

    def forward(self, x):
        atts = []
        for attention_head in self.attention_heads:
            atts.append(attention_head(x))
        atts = torch.cat(atts, dim=-1)
        out  = self.output_linear(atts)
        return out


if __name__ == '__main__':
    inpt = torch.randn(100, 5)

    sat  = AttentionL2(5, 3)
    ret  = sat(inpt)
    print(ret.shape)

    mat = MultiHeadSelfAttentionL2(5, 4, 3, spectral_scaling=True)
    ret = mat(inpt)
    print(ret.shape)
