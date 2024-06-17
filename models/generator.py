import torch
import torch.nn as nn
import functools
from torch.optim import lr_scheduler
import torchvision.models as models
import torch.nn.functional as F
from models.attention import CBAM



def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9,
                 padding_type='reflect'):
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                         output_padding=1, bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias,rate):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias,rate)
    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias,rate):
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

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias,dilation=rate),nn.ReflectionPad2d(rate-1), norm_layer(dim), nn.ReLU(True)]
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

    def forward(self, x):
        out = x + self.conv_block(x) # add skip connections
        return out





class Resnet1Block(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias,rate):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(Resnet1Block, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias,rate)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias,rate):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
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

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias,dilation=rate),nn.ReflectionPad2d(rate-1), norm_layer(dim), nn.ReLU(True)]
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
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias,dilation=1), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x) # add skip connections
        return out


class DenseBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias,num_convs):
        super(DenseBlock, self).__init__()
        self.layer=[]
        rates=[1,2,5]
        for i in range(num_convs):
            self.layer+=[self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias,rates[i])]
        self.net=nn.Sequential(*self.layer)
    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias,rate):
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

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias,dilation=rate),nn.ReflectionPad2d(rate-1), norm_layer(dim), nn.ReLU(True)]
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

    def forward(self, x):
        for densenet in self.net:
            y =densenet(x)
            x=x+y # add skip connections
        return x

class DensenetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9,
                 padding_type='reflect'):
        super(DensenetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]


        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),

                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
        rates = [1, 2, 5, 1, 2 ,5 ,1,2 ,5 ]
        mult = 2 ** n_downsampling
        #for i in range(n_blocks):  # add ResNet blocks
        num_convs_in_dense_blocks = [3, 3, 3]

        for i, num_convs in enumerate(num_convs_in_dense_blocks):
            model += [DenseBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias,num_convs=num_convs)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                         output_padding=1, bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class net3Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9,
                 padding_type='reflect'):
        super(net3Generator, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model1 = [nn.ReflectionPad2d(3),
                  nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                  norm_layer(ngf),
                  nn.ReLU(True)]


        n_downsampling = 2
        # add downsampling layers
        mult = 2 ** 0
        model2 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                       norm_layer(ngf * mult * 2),
                       nn.ReLU(True)]

        mult = 2 ** 1
        model3 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                   norm_layer(ngf * mult * 2),
                   nn.ReLU(True)]

        model4 = []
        model5 = []
        model6 = []
        mult = 2 ** n_downsampling

        # add DenseNet blocks
        model4 += [DenseBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                 use_bias=use_bias, num_convs=3)]
        # add DenseNet blocks
        model5 += [DenseBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias, num_convs=3)]
        # add DenseNet blocks
        model6 += [DenseBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                 use_bias=use_bias, num_convs=3)]


        mult = 2 ** (n_downsampling - 0)
        model7 = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                          output_padding=1, bias=use_bias),
                       norm_layer(int(ngf * mult / 2)),
                       nn.ReLU(True)]
        mult = 2 ** (n_downsampling - 1)
        model8 = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                      output_padding=1, bias=use_bias),
                   norm_layer(int(ngf * mult / 2)),
                   nn.ReLU(True)]
        model9=[]
        model9 += [nn.ReflectionPad2d(3)]
        model9 += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model9 += [nn.Tanh()]


        self.dy2=CBAM(128)
        self.dy3=CBAM(256)
        self.ca = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256 * 3, 256 // 16, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(256 // 16, 256 * 3, 1, padding=0, bias=True),
            nn.Sigmoid()
        ])
        self.pa=PALayer(channel=256)
        self.down1 = nn.Sequential(*model1)
        self.down2 = nn.Sequential(*model2)
        self.down3 = nn.Sequential(*model3)
        self.res1 = nn.Sequential(*model4)
        self.res2 = nn.Sequential(*model5)
        self.res3 = nn.Sequential(*model6)
        self.up1 = nn.Sequential(*model7)
        self.up2 = nn.Sequential(*model8)
        self.up3 = nn.Sequential(*model9)
        self.mx1 = Mix()
        self.mx2 = Mix()

    def forward(self, input):
        x_down1 = self.down1(input)
        x_down2 = self.down2(x_down1)
        x_down3 = self.down3(x_down2)
        x_res1 = self.res1(x_down3)
        x_res2=self.res2(x_res1)
        x_res3=self.res3(x_res2)
        w = self.ca(torch.cat([x_res1, x_res2, x_res3], dim=1))
        w = w.view(-1, 3, 256)[:, :, :, None, None]
        out = w[:, 0, ::] * x_res1 + w[:, 1, ::] * x_res2 + w[:, 2, ::] * x_res3
        out = self.pa(out)
        x_dy3 = self.dy3(x_down3)
        x_up1 = self.up1(self.mx1(x_dy3,out))
        x_dy2 = self.dy2(x_down2)
        x_up2 = self.up2(self.mx2(x_dy2,x_up1))
        x_up3 = self.up3(x_up2)
        return x_up3

class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        print(fea2.shape)
        print(fea1.shape)
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out

class netGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9,
                 padding_type='reflect'):
        super(netGenerator, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model1 = [nn.ReflectionPad2d(3),
                  nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                  norm_layer(ngf),
                  nn.ReLU(True)]


        n_downsampling = 2
        # add downsampling layers
        mult = 2 ** 0
        model2 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                       norm_layer(ngf * mult * 2),
                       nn.ReLU(True)]

        mult = 2 ** 1
        model3 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                   norm_layer(ngf * mult * 2),
                   nn.ReLU(True)]

        model4 = []
        model5 = []
        model6 = []
        mult = 2 ** n_downsampling

        # add DenseNet blocks
        model4 += [DenseBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                 use_bias=use_bias, num_convs=3)]
        # add DenseNet blocks
        model5 += [DenseBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias, num_convs=3)]
        # add DenseNet blocks
        model6 += [DenseBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                 use_bias=use_bias, num_convs=3)]


        mult = 2 ** (n_downsampling - 0)
        model7 = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                          output_padding=1, bias=use_bias),
                       norm_layer(int(ngf * mult / 2)),
                       nn.ReLU(True)]
        mult = 2 ** (n_downsampling - 1)
        model8 = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                      output_padding=1, bias=use_bias),
                   norm_layer(int(ngf * mult / 2)),
                   nn.ReLU(True)]
        model9=[]
        model9 += [nn.ReflectionPad2d(3)]
        model9 += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model9 += [nn.Tanh()]

        self.dy1=DYModule(64,64)
        self.dy2=DYModule(128,128)
        self.dy3 = DYModule(256, 256)
        self.ca = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256 * 3, 256 // 16, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(256 // 16, 256 * 3, 1, padding=0, bias=True),
            nn.Sigmoid()
        ])
        self.pa=PALayer(channel=256)
        self.down1 = nn.Sequential(*model1)
        self.down2 = nn.Sequential(*model2)
        self.down3 = nn.Sequential(*model3)
        self.res1 = nn.Sequential(*model4)
        self.res2 = nn.Sequential(*model5)
        self.res3 = nn.Sequential(*model6)
        self.up1 = nn.Sequential(*model7)
        self.up2 = nn.Sequential(*model8)
        self.up3 = nn.Sequential(*model9)

        self.mix1 = Mix(m=-1)
        self.mix2 = Mix(m=-0.8)
        self.mix3 = Mix(m=-0.6)

    def forward(self, input):
        x_down1 = self.down1(input)
        x_down2 = self.down2(x_down1)
        x_down3 = self.down3(x_down2)
        x_res1 = self.res1(x_down3)
        x_res2=self.res2(x_res1)
        x_res3=self.res3(x_res2)
        w = self.ca(torch.cat([x_res1, x_res2, x_res3], dim=1))
        w = w.view(-1, 3, 256)[:, :, :, None, None]
        out = w[:, 0, ::] * x_res1 + w[:, 1, ::] * x_res2 + w[:, 2, ::] * x_res3
        out = self.pa(out)
        x_dy3 = self.dy3(x_down3)
        x_up1 = self.up1(self.mix1(x_dy3,out))
        x_dy2 = self.dy2(x_down2)
        x_up2 = self.up2(self.mix2(x_dy2,x_up1))
        x_dy1 = self.dy1(x_down1)
        x_up3 = self.up3(self.mix3(x_dy1,x_up2))
        return x_up3

class net2Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9,
                 padding_type='reflect'):
        super(net2Generator, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model1 = [nn.ReflectionPad2d(3),
                  nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                  norm_layer(ngf),
                  nn.ReLU(True)]


        n_downsampling = 2
        # add downsampling layers
        mult = 2 ** 0
        model2 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                       norm_layer(ngf * mult * 2),
                       nn.ReLU(True)]

        mult = 2 ** 1
        model3 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                   norm_layer(ngf * mult * 2),
                   nn.ReLU(True)]

        model4 = []
        model5 = []
        model6 = []
        mult = 2 ** n_downsampling

        # add DenseNet blocks
        model4 += [DenseBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                 use_bias=use_bias, num_convs=3)]
        # add DenseNet blocks
        model5 += [DenseBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias, num_convs=3)]
        # add DenseNet blocks
        model6 += [DenseBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                 use_bias=use_bias, num_convs=3)]


        mult = 2 ** (n_downsampling - 0)
        model7 = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                          output_padding=1, bias=use_bias),
                       norm_layer(int(ngf * mult / 2)),
                       nn.ReLU(True)]
        mult = 2 ** (n_downsampling - 1)
        model8 = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                      output_padding=1, bias=use_bias),
                   norm_layer(int(ngf * mult / 2)),
                   nn.ReLU(True)]
        model9=[]
        model9 += [nn.ReflectionPad2d(3)]
        model9 += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model9 += [nn.Tanh()]
        self.ca = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256 * 3, 256 // 16, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(256 // 16, 256 * 3, 1, padding=0, bias=True),
            nn.Sigmoid()
        ])
        self.pa=PALayer(channel=256)
        self.down1 = nn.Sequential(*model1)
        self.down2 = nn.Sequential(*model2)
        self.down3 = nn.Sequential(*model3)
        self.res1 = nn.Sequential(*model4)
        self.res2 = nn.Sequential(*model5)
        self.res3 = nn.Sequential(*model6)
        self.up1 = nn.Sequential(*model7)
        self.up2 = nn.Sequential(*model8)
        self.up3 = nn.Sequential(*model9)
        self.mx1 = Mix()
        self.mx2 = Mix()

    def forward(self, input):
        x_down1 = self.down1(input)
        x_down2 = self.down2(x_down1)
        x_down3 = self.down3(x_down2)
        x_res1 = self.res1(x_down3)
        x_res2=self.res2(x_res1)
        x_res3=self.res3(x_res2)
        w = self.ca(torch.cat([x_res1, x_res2, x_res3], dim=1))
        w = w.view(-1, 3, 256)[:, :, :, None, None]
        out = w[:, 0, ::] * x_res1 + w[:, 1, ::] * x_res2 + w[:, 2, ::] * x_res3
        out = self.pa(out)
        x_up1 = self.up1(self.mx1(x_down3,out))
        x_up2 = self.up2(self.mx2(x_down2,x_up1))
        x_up3 = self.up3(x_up2)
        return x_up3


class net1Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9,
                 padding_type='reflect'):
        super(net1Generator, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model1 = [nn.ReflectionPad2d(3),
                  nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                  norm_layer(ngf),
                  nn.ReLU(True)]


        n_downsampling = 2
        # add downsampling layers
        mult = 2 ** 0
        model2 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                       norm_layer(ngf * mult * 2),
                       nn.ReLU(True)]

        mult = 2 ** 1
        model3 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                   norm_layer(ngf * mult * 2),
                   nn.ReLU(True)]

        model4 = []
        model5 = []
        model6 = []
        mult = 2 ** n_downsampling

        # add DenseNet blocks
        model4 += [DenseBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                 use_bias=use_bias, num_convs=3)]
        # add DenseNet blocks
        model5 += [DenseBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias, num_convs=3)]
        # add DenseNet blocks
        model6 += [DenseBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                 use_bias=use_bias, num_convs=3)]


        mult = 2 ** (n_downsampling - 0)
        model7 = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                          output_padding=1, bias=use_bias),
                       norm_layer(int(ngf * mult / 2)),
                       nn.ReLU(True)]
        mult = 2 ** (n_downsampling - 1)
        model8 = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                      output_padding=1, bias=use_bias),
                   norm_layer(int(ngf * mult / 2)),
                   nn.ReLU(True)]
        model9=[]
        model9 += [nn.ReflectionPad2d(3)]
        model9 += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model9 += [nn.Tanh()]

        self.dy1=DYModule(64,64)
        self.dy2=DYModule(128,128)
        self.dy3 = DYModule(256, 256)
        self.ca = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256 * 3, 256 // 16, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(256 // 16, 256 * 3, 1, padding=0, bias=True),
            nn.Sigmoid()
        ])
        self.pa=PALayer(channel=256)
        self.down1 = nn.Sequential(*model1)
        self.down2 = nn.Sequential(*model2)
        self.down3 = nn.Sequential(*model3)
        self.res1 = nn.Sequential(*model4)
        self.res2 = nn.Sequential(*model5)
        self.res3 = nn.Sequential(*model6)
        self.up1 = nn.Sequential(*model7)
        self.up2 = nn.Sequential(*model8)
        self.up3 = nn.Sequential(*model9)


    def forward(self, input):
        x_down1 = self.down1(input)
        x_down2 = self.down2(x_down1)
        x_down3 = self.down3(x_down2)
        x_res1 = self.res1(x_down3)
        x_res2=self.res2(x_res1)
        x_res3=self.res3(x_res2)
        w = self.ca(torch.cat([x_res1, x_res2, x_res3], dim=1))
        w = w.view(-1, 3, 256)[:, :, :, None, None]
        out = w[:, 0, ::] * x_res1 + w[:, 1, ::] * x_res2 + w[:, 2, ::] * x_res3
        out = self.pa(out)
        x_dy3 = self.dy3(x_down3)
        x_up1 = self.up1(x_dy3*0.2+out*0.8)
        x_dy2 = self.dy2(x_down2)
        x_up2 = self.up2(x_dy2*0.2+x_up1*0.8)
        x_dy1 = self.dy1(x_down1)
        x_up3 = self.up3(x_dy1*0.2+x_up2*0.8)
        return x_up3

class attention2d(nn.Module):
    def __init__(self, in_planes, K,):
        super(attention2d, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, K, 1,)
        self.fc2 = nn.Conv2d(K, K, 1,)
    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x, 1)

class netrGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9,
                 padding_type='reflect'):
        super(netrGenerator, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model1 = [nn.ReflectionPad2d(3),
                  nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                  norm_layer(ngf),
                  nn.ReLU(True)]


        n_downsampling = 2
        # add downsampling layers
        mult = 2 ** 0
        model2 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                       norm_layer(ngf * mult * 2),
                       nn.ReLU(True)]

        mult = 2 ** 1
        model3 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                   norm_layer(ngf * mult * 2),
                   nn.ReLU(True)]

        model4 = []
        model5 = []
        model6 = []
        mult = 2 ** n_downsampling

        # add DenseNet blocks
        model4 += [DenseBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                 use_bias=use_bias, num_convs=3)]
        # add DenseNet blocks
        model5 += [DenseBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias, num_convs=3)]
        # add DenseNet blocks
        model6 += [DenseBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                 use_bias=use_bias, num_convs=3)]


        mult = 2 ** (n_downsampling - 0)
        model7 = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                          output_padding=1, bias=use_bias),
                       norm_layer(int(ngf * mult / 2)),
                       nn.ReLU(True)]
        mult = 2 ** (n_downsampling - 1)
        model8 = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                      output_padding=1, bias=use_bias),
                   norm_layer(int(ngf * mult / 2)),
                   nn.ReLU(True)]
        model9=[]
        model9 += [nn.ReflectionPad2d(3)]
        model9 += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model9 += [nn.Tanh()]

        self.ca = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256 * 3, 256 // 16, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(256 // 16, 256 * 3, 1, padding=0, bias=True),
            nn.Sigmoid()
        ])
        self.pa=PALayer(channel=256)
        self.down1 = nn.Sequential(*model1)
        self.down2 = nn.Sequential(*model2)
        self.down3 = nn.Sequential(*model3)
        self.res1 = nn.Sequential(*model4)
        self.res2 = nn.Sequential(*model5)
        self.res3 = nn.Sequential(*model6)
        self.up1 = nn.Sequential(*model7)
        self.up2 = nn.Sequential(*model8)
        self.up3 = nn.Sequential(*model9)


    def forward(self, input):
        x_down1 = self.down1(input)
        x_down2 = self.down2(x_down1)
        x_down3 = self.down3(x_down2)
        x_res1 = self.res1(x_down3)
        x_res2=self.res2(x_res1)
        x_res3=self.res3(x_res2)
        w = self.ca(torch.cat([x_res1, x_res2, x_res3], dim=1))
        w = w.view(-1, 3, 256)[:, :, :, None, None]
        out = w[:, 0, ::] * x_res1 + w[:, 1, ::] * x_res2 + w[:, 2, ::] * x_res3
        out = self.pa(out)
        x_up1 = self.up1(x_down3*0.2+out*0.8)
        x_up2 = self.up2(x_down2*0.2+x_up1*0.8)
        x_up3 = self.up3(x_down1*0.2+x_up2*0.8)
        return x_up3


class netaGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9,
                 padding_type='reflect'):
        super(netaGenerator, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model1 = [nn.ReflectionPad2d(3),
                  nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                  norm_layer(ngf),
                  nn.ReLU(True)]


        n_downsampling = 2
        # add downsampling layers
        mult = 2 ** 0
        model2 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                       norm_layer(ngf * mult * 2),
                       nn.ReLU(True)]

        mult = 2 ** 1
        model3 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                   norm_layer(ngf * mult * 2),
                   nn.ReLU(True)]

        model4 = []
        model5 = []
        model6 = []
        mult = 2 ** n_downsampling

        # add DenseNet blocks
        model4 += [DenseBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                 use_bias=use_bias, num_convs=3)]
        # add DenseNet blocks
        model5 += [DenseBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias, num_convs=3)]
        # add DenseNet blocks
        model6 += [DenseBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                 use_bias=use_bias, num_convs=3)]


        mult = 2 ** (n_downsampling - 0)
        model7 = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                          output_padding=1, bias=use_bias),
                       norm_layer(int(ngf * mult / 2)),
                       nn.ReLU(True)]
        mult = 2 ** (n_downsampling - 1)
        model8 = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                      output_padding=1, bias=use_bias),
                   norm_layer(int(ngf * mult / 2)),
                   nn.ReLU(True)]
        model9=[]
        model9 += [nn.ReflectionPad2d(3)]
        model9 += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model9 += [nn.Tanh()]

        self.ca = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256 * 3, 256 // 16, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(256 // 16, 256 * 3, 1, padding=0, bias=True),
            nn.Sigmoid()
        ])
        self.pa=PALayer(channel=256)
        self.down1 = nn.Sequential(*model1)
        self.down2 = nn.Sequential(*model2)
        self.down3 = nn.Sequential(*model3)
        self.res1 = nn.Sequential(*model4)
        self.res2 = nn.Sequential(*model5)
        self.res3 = nn.Sequential(*model6)
        self.up1 = nn.Sequential(*model7)
        self.up2 = nn.Sequential(*model8)
        self.up3 = nn.Sequential(*model9)


    def forward(self, input):
        x_down1 = self.down1(input)
        x_down2 = self.down2(x_down1)
        x_down3 = self.down3(x_down2)
        x_res1 = self.res1(x_down3)
        x_res2=self.res2(x_res1)
        x_res3=self.res3(x_res2)
        w = self.ca(torch.cat([x_res1, x_res2, x_res3], dim=1))
        w = w.view(-1, 3, 256)[:, :, :, None, None]
        out = w[:, 0, ::] * x_res1 + w[:, 1, ::] * x_res2 + w[:, 2, ::] * x_res3
        out = self.pa(out)
        x_up1 = self.up1(out)
        x_up2 = self.up2(x_up1)
        x_up3 = self.up3(x_up2)
        return x_up3

class CAM(nn.Module):
    '''
    单特征 进行通道加权,作用类似SE模块
    '''

    def __init__(self, channels=64, r=4):
        super(CAM, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return x * wei

class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.BatchNorm2d(channel // 8),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class SEModule_small(nn.Module):
    def __init__(self, channel):
        super(SEModule_small, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel, bias=False),
            Hsigmoid()
        )

    def forward(self, x):
        y = self.fc(x)
        return x * y



class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 3.

class DYModule(nn.Module):
    def __init__(self, inp, oup, fc_squeeze=8):
        super(DYModule, self).__init__()
        self.conv = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)
        if inp < oup:
            self.mul = 4
            reduction = 8
            self.avg_pool = nn.AdaptiveAvgPool2d(2)
        else:
            self.mul = 1
            reduction = 2
            self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.dim = min((inp * self.mul) // reduction, oup // reduction)
        while self.dim ** 2 > inp * self.mul * 2:
            reduction *= 2
            self.dim = min((inp * self.mul) // reduction, oup // reduction)
        if self.dim < 4:
            self.dim = 4

        squeeze = max(inp * self.mul, self.dim ** 2) // fc_squeeze
        if squeeze < 4:
            squeeze = 4
        self.conv_q = nn.Conv2d(inp, self.dim, 1, 1, 0, bias=False)

        self.fc = nn.Sequential(
            nn.Linear(inp * self.mul, squeeze, bias=False),
            SEModule_small(squeeze),
        )

        self.attention = CAM(inp * self.mul)

        self.fc_phi = nn.Linear(squeeze, self.dim ** 2, bias=False)
        self.fc_scale = nn.Linear(squeeze, oup, bias=False)
        self.hs = Hsigmoid()
        self.conv_p = nn.Conv2d(self.dim, oup, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(self.dim)
        self.bn2 = nn.BatchNorm1d(self.dim)

    def forward(self, x):
        r = self.conv(x)

        b, c, h, w = x.size()
        z = self.attention(x)
        y = self.avg_pool(z).view(b, c * self.mul)
        y = self.fc(y)

        dy_phi = self.fc_phi(y).view(b, self.dim, self.dim)
        dy_scale = self.hs(self.fc_scale(y)).view(b, -1, 1, 1)
        r = dy_scale.expand_as(r) * r

        x = self.conv_q(x)
        x = self.bn1(x)
        x = x.view(b, -1, h * w)
        x = self.bn2(torch.matmul(dy_phi, x)) + x
        x = x.view(b, -1, h, w)
        x = self.conv_p(x)
        return x + r






def test():
    x = torch.randn([4, 3, 512, 640])
    y= torch.randn([4, 3, 256, 256])

    gen =net3Generator(3,3)
    print(gen)
    z = gen(x)
    print(z.size())




if __name__ == "__main__":
    test()