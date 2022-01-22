import torch
import torch.nn as nn


class AttentionL2(nn.Module):
    def __init__(self, in_features, out_features, scale=None):
        """
        Single head L2-attention module
        :param in_features: number of input features
        :param out_features: number of out features
        """
        super(AttentionL2, self).__init__()

        self.in_features  = in_features
        self.out_features = out_features
        self.scale        = out_features if scale is None else scale

        self.q = nn.Linear(self.in_features, self.out_features, bias=False)
        self.k = nn.Linear(self.in_features, self.out_features, bias=False)
        self.v = nn.Linear(self.in_features, self.out_features, bias=False)

        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        l2_dist = torch.cdist(q, k, p=2)
        att = self.softmax(l2_dist / (self.scale**(1/2))) @ v
        return att


class MultiHeadSelfAttentionL2(nn.Module):
    def __init__(self, in_features, n_head, head_dim, output_size=None):
        """
        Multihead self L2-attention module based on L2-attention module
        :param in_features: number of input features
        :param n_head: number of attention heads
        :param head_dim: output size of each attention head
        :param output_size: final output feature number, default is n_head * head_dim
        """
        super(MultiHeadSelfAttentionL2, self).__init__()

        self.out_dim         = n_head * head_dim
        self.out_features    = self.out_dim if output_size is None else output_size

        self.attention_heads = nn.ModuleList([AttentionL2(in_features, head_dim, scale=self.out_dim) for i in range(n_head)])
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

    att  = AttentionL2(5, 3)
    ret  = att(inpt)
    print(ret.shape)

    mat = MultiHeadSelfAttentionL2(5, 4, 3)
    ret = mat(inpt)
    print(ret.shape)
