import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, in_features, out_features, scale=None, spectral_scaling=False, lp=2, **kwargs):
        """
        Single head L2-attention module
        :param in_features: number of input features
        :param out_features: number of out features
        :param scale: rescale factor of d(K,Q), default is out_features
        :param spectral_scaling: perform spectral rescaling of q, k, v linear layers at each forward call
        :param lp: norm used for attention, should be 1 or 2, default 2
        """
        super(Attention, self).__init__()

        self.in_features      = in_features
        self.out_features     = out_features
        self.scale            = out_features if scale is None else scale
        self.spectral_scaling = spectral_scaling
        self.lp               = lp

        assert lp in [1, 2], f'Unsupported norm for attention: lp={lp} but should be 1 or 2'
        self.attention_func = self._l1att if self.lp == 1 else self._l2att

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

        att = self.attention_func(q, k)  # we use L2 reg only in discriminator
        att = self.softmax(att / (self.scale**(1/2))) @ v
        return att

    def _get_spectrum(self):
        _, sq, _ = torch.svd(self.q.weight)
        _, sk, _ = torch.svd(self.k.weight)
        _, sv, _ = torch.svd(self.v.weight)
        return sq, sk, sv

    def _weight_spectral_rescale(self):
        sq, sk, sv = self._get_spectrum()
        self.q.weight = nn.Parameter(self.init_spectrum[0] / max(sq) * self.q.weight)
        self.k.weight = nn.Parameter(self.init_spectrum[1] / max(sk) * self.k.weight)
        self.v.weight = nn.Parameter(self.init_spectrum[2] / max(sv) * self.v.weight)

    def _l2att(self, q, k):
        return torch.cdist(q, k, p=2)

    def _l1att(self, q, k):
        return torch.einsum("... i d, ... j d -> ... i j", q, k)


class MultiHeadSelfAttention(nn.Module):
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
        super(MultiHeadSelfAttention, self).__init__()

        self.out_dim         = n_head * head_dim
        self.out_features    = self.out_dim if output_size is None else output_size

        self.attention_heads = nn.ModuleList([Attention(in_features, head_dim, scale=self.out_dim, spectral_scaling=spectral_scaling, lp=lp) for _ in range(n_head)])
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

    sat  = Attention(5, 3)
    ret  = sat(inpt)
    print(ret.shape)

    mat = MultiHeadSelfAttention(5, 4, 3, spectral_scaling=True)
    ret = mat(inpt)
    print(ret.shape)
