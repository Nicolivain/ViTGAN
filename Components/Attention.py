import torch
import torch.nn as nn
from einops import rearrange, repeat


class AttentionL2(nn.Module):
    def __init__(self, in_features, out_features, scale=None, spectral_scaling=False, **kwargs):
        """
        Single head L2-attention module
        :param in_features: number of input features
        :param out_features: number of out features
        :param scale: rescale factor of d(K,Q), default is out_features
        :param spectral_scaling: perform spectral rescaling of q, k, v linear layers at each forward call
        """
        super(AttentionL2, self).__init__()

        self.in_features      = in_features
        self.out_features     = out_features
        self.scale            = out_features if scale is None else scale
        self.spectral_scaling = spectral_scaling

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

        l2_dist = torch.cdist(q, k, p=2) if self.spectral_scaling else q @ v  # we use L2 reg only in discriminator
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
    def __init__(self, in_features, n_head, head_dim, output_size=None, spectral_scaling=False, **kwargs):
        """
        Multihead self L2-attention module based on L2-attention module
        :param in_features: number of input features
        :param n_head: number of attention heads
        :param head_dim: output size of each attention head
        :param output_size: final output feature number, default is n_head * head_dim
        :param spectral_scaling: perform spectral rescaling of q, k, v for each head
        """
        super(MultiHeadSelfAttentionL2, self).__init__()

        self.out_dim         = n_head * head_dim
        self.out_features    = self.out_dim if output_size is None else output_size

        self.attention_heads = nn.ModuleList([AttentionL2(in_features, head_dim, scale=self.out_dim, spectral_scaling=spectral_scaling) for _ in range(n_head)])
        self.output_linear   = nn.Linear(self.out_dim, self.out_features)

    def forward(self, x):
        atts = []
        for attention_head in self.attention_heads:
            atts.append(attention_head(x))
        atts = torch.cat(atts, dim=-1)
        out  = self.output_linear(atts)
        return out


class ScrappedAttention(nn.Module):
    """
    Implement multi head self attention layer using the "Einstein summation convention".

    Parameters
    ----------
    in_features:
        Token's dimension, EX: word embedding vector size
    n_head:
        The number of distinct representations to learn
    head_dim:
        The dimension of the each head
    discriminator:
        Used in discriminator or not.
    """
    def __init__(self, in_features, n_head = 4, head_dim = None, output_size=None, spectral_scaling = False):
        super().__init__()
        self.n_head = n_head
        self.outsize = in_features if output_size is None else output_size
        self.head_dim = int(in_features / n_head) if head_dim is None else head_dim
        self.weight_dim = self.n_head * self.head_dim
        self.to_qkv = nn.Linear(in_features, self.weight_dim * 3, bias = False)
        self.scale_factor = in_features ** -0.5
        self.spectral_scaling = spectral_scaling
        self.w_out = nn.Linear(self.weight_dim, self.outsize, bias = True)

        if spectral_scaling:
            u, s, v = torch.svd(self.to_qkv.weight)
            self.init_spect_norm = torch.max(s)

    def forward(self, x):
        assert x.dim() == 3

        if self.spectral_scaling:
            u, s, v = torch.svd(self.to_qkv.weight)
            self.to_qkv.weight = torch.nn.Parameter(self.to_qkv.weight * self.init_spect_norm / torch.max(s))

        # Generate the q, k, v vectors
        qkv = self.to_qkv(x)
        q, k, v = tuple(rearrange(qkv, 'b t (d k h) -> k b h t d', k = 3, h = self.n_head))

        # Enforcing Lipschitzness of Transformer Discriminator
        # Due to Lipschitz constant of standard dot product self-attention
        # layer can be unbounded, so adopt the l2 attention replace the dot product.
        if self.spectral_scaling:
            attn = torch.cdist(q, k, p = 2)
        else:
            attn = torch.einsum("... i d, ... j d -> ... i j", q, k)
        scale_attn = attn * self.scale_factor
        scale_attn_score = torch.softmax(scale_attn, dim = -1)
        result = torch.einsum("... i j, ... j d -> ... i d", scale_attn_score, v)

        # re-compose
        result = rearrange(result, "b h t d -> b t (h d)")
        return self.w_out(result)


if __name__ == '__main__':
    inpt = torch.randn(100, 5)

    sat  = AttentionL2(5, 3)
    ret  = sat(inpt)
    print(ret.shape)

    mat = MultiHeadSelfAttentionL2(5, 4, 3, spectral_scaling=True)
    ret = mat(inpt)
    print(ret.shape)
