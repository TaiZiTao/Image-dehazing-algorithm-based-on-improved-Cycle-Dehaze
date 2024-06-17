import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict


def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
    conv_block = []
    p = 0
    if padding_type == 'reflect':
        conv_block += [nn.ReflectionPad2d(1)]
    elif padding_type == 'replicate':
        conv_block += [nn.ReplicationPad2d(1)]
    elif padding_type == 'zero':
        p = 1
    else:
        raise NotImplementedError('padding [%s] is not implemented' % padding_type)

    conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
    if use_dropout:
        conv_block += [nn.Dropout(0.5)]

    p = 0
    if padding_type == 'reflect':
        conv_block += [nn.ReflectionPad2d(1)]
    elif padding_type == 'replicate':
        conv_block += [nn.ReplicationPad2d(1)]
    elif padding_type == 'zero':
        p = 1
    else:
        raise NotImplementedError('padding [%s] is not implemented' % padding_type)
    conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

    return nn.Sequential(*conv_block)

def conv_block(input_channels, output_channels,use_bias):
    # 一般 1 x 1 卷积的通道数为 growthRate 的 4 倍
    inter_channels =  output_channels

    return nn.Sequential(
        nn.Conv2d(input_channels, inter_channels, kernel_size=3, padding=1,bias=use_bias),
        nn.BatchNorm2d(inter_channels),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Conv2d(inter_channels, output_channels, kernel_size=3, padding=1, bias=use_bias),
        nn.BatchNorm2d(inter_channels),
        nn.ReLU(inplace=True)
    )


# output_channels 也被称为 growthRate
class DenseBlock(nn.Module):
    def __init__(self, num_convs, dim,use_bias):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(dim, dim,use_bias))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # Concatenate the input and output of each block on the channel
            # dimension
            X = X+Y
        return X

class DenseNet(nn.Module):
    def __init__(self, dim,use_bias):
        super(DenseNet, self).__init__()
        num_convs_in_dense_blocks = [3,3,3]
        blks = []
        for i, num_convs in enumerate(num_convs_in_dense_blocks):
            blks.append(DenseBlock(num_convs, dim,use_bias))
        self.blks = nn.Sequential(*blks)


    def forward(self, x):
        y = self.blks(x)
        return y


def test():
    x = torch.randn([1, 256, 64, 64])
    print(x.size())
    gen = DenseNet(dim=256,use_bias=True)
    print(gen)
    y = gen(x)
    print(y.size())



if __name__ == "__main__":
    test()