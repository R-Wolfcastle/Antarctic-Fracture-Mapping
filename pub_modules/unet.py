import torch
import torch.nn as nn
import torchvision.transforms.functional as tf
from random import randrange


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.double_conv_1 = self.double_conv_initial(in_channels, 32, 7, 3)
        self.down_conv_1 = self.double_conv_down(32, 64, 3, 1)
        self.down_conv_2 = self.double_conv_down(64, 128, 3, 1)
        self.bottle = self.bottle_block(128, 256, 128, 3, 1)

        self.up_conv_1 = self.double_conv_up(128*2, 128, 64, 3, 1)
        self.up_conv_2 = self.double_conv_up(64*2, 64, 32, 3, 1)

        self.out = self.triple_conv_final(32*2, 32, out_channels, 3, 1)

    def __call__(self, x):
        # print(x.shape)
        c1 = self.double_conv_1(x)
        # print(c1.shape)
        c2 = self.down_conv_1(c1)
        # print(c2.shape)
        c3 = self.down_conv_2(c2)
        # print(c3.shape)

        b1 = self.bottle(c3)
        # print(b1.shape)

        up1 = self.up_conv_1(torch.cat([b1, c3], 1))
        # print(up1.shape)
        up2 = self.up_conv_2(torch.cat([up1, c2], 1))
        # print(up2.shape)

        out = self.out(torch.cat([up2, c1], 1))
        # print(out.shape)

        # raise

        return out


    def double_conv_initial(self, in_channels, out_channels, kernel_size, padding):
        dc = nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, padding_mode='reflect'),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Dropout(0.5),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, padding_mode='reflect'),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Dropout(0.5)
        )
        return dc

    def triple_conv_final(self, in_channels, int_channels, out_channels, kernel_size, padding):
        tcf = nn.Sequential(
            torch.nn.Conv2d(in_channels, int_channels, kernel_size=kernel_size, stride=1, padding=padding, padding_mode='reflect'),
            torch.nn.BatchNorm2d(int_channels),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Dropout(0.5),
            torch.nn.Conv2d(int_channels, int_channels, kernel_size=kernel_size, stride=1, padding=padding, padding_mode='reflect'),
            torch.nn.BatchNorm2d(int_channels),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Dropout(0.5),
            torch.nn.Conv2d(int_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, padding_mode='reflect')
        )
        return tcf

    def double_conv_down(self, in_channels, out_channels, kernel_size, padding):
        dcd = nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, padding_mode='reflect'),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Dropout(0.5),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, padding_mode='reflect'),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Dropout(0.5)
        )
        return dcd

    def double_conv_up(self, in_channels, int_channels, out_channels, kernel_size, padding):
        dcu = nn.Sequential(
            torch.nn.Conv2d(in_channels, int_channels, kernel_size=kernel_size, stride=1, padding=padding, padding_mode='reflect'),
            torch.nn.BatchNorm2d(int_channels),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Dropout(0.5),
            torch.nn.Conv2d(int_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, padding_mode='reflect'),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Dropout(0.5),
            # torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
            torch.nn.Upsample(scale_factor=2, mode='bilinear')
        )
        return dcu

    def bottle_block(self, in_channels, int_channels, out_channels, kernel_size, padding):
        bottle = nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            torch.nn.Conv2d(in_channels, int_channels, kernel_size, stride=1, padding=padding, padding_mode='reflect'),
            torch.nn.BatchNorm2d(int_channels),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Dropout(0.5),
            torch.nn.Conv2d(int_channels, out_channels, kernel_size, stride=1, padding=padding, padding_mode='reflect'),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Dropout(0.5),
            # torch.nn.ConvTranspose2d(int_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
            torch.nn.Upsample(scale_factor=2, mode='bilinear')
        )
        return bottle
