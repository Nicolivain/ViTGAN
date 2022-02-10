import torch
import torch.nn as nn


class PatchEncoder(nn.Module):
    def __init__(self, img_size, n_channels, patch_size=8, projection_ouput_size=None, overlap=2, dropout_rate=0.0, **kwargs):
        """
        Encodes an image to a vector according to ViT process: patches, projection, cls token and positional embedding
        :param img_size: input images size, the image must be square sized
        :param n_channels: number of channel in the input images
        :param patch_size: size of each patches, patches will be square sized
        :param projection_ouput_size: number of feature at the output of the projection
        :param overlap: number of overlapping pixels for neighbouring patches
        :param dropout_rate: dropout rate at the final stage level
        """
        super(PatchEncoder, self).__init__()

        self.patch_size       = patch_size
        self.overlap          = overlap

        self.token_size = n_channels * (self.patch_size + 2 * self.overlap)**2
        self.stride     = (img_size - self.patch_size - 2 * self.overlap) // self.patch_size + 1
        self.n_token    = ((img_size - (self.patch_size + 2 * self.overlap-1) - 1) // self.stride + 1)**2

        self.proj_output_size = projection_ouput_size if projection_ouput_size is not None else self.token_size
        self.projection_matrix = nn.Linear(self.token_size, self.proj_output_size, bias=False)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.proj_output_size))
        self.pos_embedding = nn.Parameter(torch.randn(self.n_token + 1, self.proj_output_size))

        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, imgs):
        assert len(imgs.shape) == 4, 'Expected input image tensor to be of shape BxCxHxW'
        imgs_tokens = self._get_tokens(imgs)
        imgs_tokens = self.projection_matrix(imgs_tokens)
        cls_token = self.cls_token.expand(imgs_tokens.shape[0], 1, self.proj_output_size)

        tokens = torch.cat((cls_token, imgs_tokens), dim=1)
        emb_tokens = tokens + self.pos_embedding
        emb_tokens = self.dropout(emb_tokens)
        return emb_tokens

    def _get_tokens(self, imgs):
        # To get the stride with overlap, we simulate a bigger patch, but accomodate only for a smaller one
        assert imgs.shape[2] == imgs.shape[3], 'The provided images are not square shaped'

        # Then we use the unfold method twice to get the batches with respect to both image dimension
        imgs_patches = imgs.unfold(2, self.patch_size + 2 * self.overlap, self.stride).unfold(3, self.patch_size + 2 * self.overlap, self.stride)

        # We reshape this 5d tensor according in a BxLxF shape: we flatten patch and channel, and flatten their 2d positionning too
        # view as Batch size, n_batch_x * n_batch_y, channels * patch_h * patch_w
        # print(imgs_patches.shape)
        # TODO: make sure the data is rearranged correctly when dimensions are ambiguous (see example below)
        imgs_patches = imgs_patches.contiguous()
        seq_patches = imgs_patches.view(imgs_patches.shape[0], imgs_patches.shape[2]*imgs_patches.shape[3], imgs_patches.shape[1]*imgs_patches.shape[4]*imgs_patches.shape[5])
        # print(seq_patches.shape)
        return seq_patches


if __name__ == '__main__':
    fim = torch.randn(100, 3, 10, 10)
    e = PatchEncoder(10, 3, 3, 5, 1)
    e.forward(fim)
