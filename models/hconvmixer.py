import torch.nn as nn
import torch.nn.functional as F
import torch
from timm.models.layers import DropPath

class Downsample(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv = nn.Conv2d(dim, 2*dim, kernel_size=1)
        self.norm = LayerNorm(2*dim, data_format="channels_first")

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = self.norm(F.gelu(x))

        return x



class Block(nn.Module):

    def __init__(self, dim, kernel_size, drop_path=0.,layer_scale_init_value=1e-6,r=1):
        super().__init__()
        self.pd = getsamepadding(kernel_size)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=self.pd, groups=dim)
        self.norm1 = LayerNorm(dim, data_format="channels_first")
        self.pwconv1 = nn.Conv2d(dim, r*dim, kernel_size=1, stride=1)
        self.act = nn.GELU()
        self.norm2 = LayerNorm(r*dim, data_format='channels_first')
        self.pwconv2 = nn.Conv2d(r*dim, dim, kernel_size=1, stride=1)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.r =r

    def forward(self, x):
        input = x
        if self.r != 1:
            x = self.dwconv(x)
            x = self.norm1(x)
            x = self.act(x)
            x = self.pwconv1(x)
            x = self.norm2(x)
            x = self.act(x)
            x = self.pwconv2(x)
            x = self.norm1(x)
            x = self.act(x)
            x += input
        else:
            x = self.dwconv(x)
            x = self.norm1(x)
            x = self.act(x)
            x += input
            x = self.pwconv1(x)
            x = self.norm1(x)
            x = self.act(x)
        return x


class basiclayer(nn.Module):
    def __init__(self, dim, kernel_size, depth, downsample=None, r=1):
        super().__init__()
        self.dim = dim
        self.depth = depth
        if downsample is not None:
            self.downsample = Downsample(dim)
        else:
            self.downsample = None

        self.block = Block(dim=dim, kernel_size=kernel_size, r=r)

    def forward(self, x):
        for i in range(self.depth):
            x = self.block(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class HConvMixer(nn.Module):

    def __init__(self, embed_dim, patch_size, kernel_size, n_classes, depth, r):
        super().__init__()
        self.patchembed = nn.Sequential(nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size),
                                        nn.GELU(),
                                        LayerNorm(embed_dim, data_format="channels_first"))
        self.n_layers = len(depth)
        self.num_features = int(embed_dim * 2 ** (self.n_layers - 1))
        self.layers = nn.ModuleList()
        for i_layer in range(self.n_layers):
            layer = basiclayer(dim=int(embed_dim * 2 ** i_layer),
                               kernel_size=kernel_size[i_layer],
                               depth=depth[i_layer],
                               downsample=Downsample if (i_layer < self.n_layers - 1) else None,
                               r=r[i_layer]
                               )
            self.layers.append(layer)
        self.pooling =nn.AdaptiveAvgPool2d((1, 1))
        self.flt = nn.Flatten()
        self.linear = nn.Linear(self.num_features, n_classes)

    def forward(self, x):
        x = self.patchembed(x)
        # print(x.size())
        for layer in self.layers:
            x = layer(x)
            # print(x.size())
        x = self.pooling(x)
        x = self.flt(x)
        x = self.linear(x)
        return x


class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.cls_vec = nn.Parameter(torch.randn(in_dim))
        self.fc = nn.Linear(in_dim, in_dim)
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        weights = torch.matmul(x.view(-1, x.shape[1]), self.cls_vec)
        weights = self.softmax(weights.view(x.shape[0], -1))
        x = torch.bmm(x.view(x.shape[0], x.shape[1], -1), weights.unsqueeze(-1)).squeeze()
        x = x + self.cls_vec
        x = self.fc(x)
        x = x + self.cls_vec
        return x

def getsamepadding(kernel_size):
    pd = (kernel_size - 1) // 2
    return pd
