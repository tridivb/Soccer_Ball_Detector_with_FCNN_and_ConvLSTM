import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.ConvLSTM import ConvLSTM

class SweatyNet2(nn.Module):
    def __init__(self, device):
        super(SweatyNet2, self).__init__()
        self.seq1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.seq2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.seq3 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.seq4 = nn.Sequential(
            nn.Conv2d(in_channels=56, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        self.seq5 = nn.Sequential(
            nn.Conv2d(in_channels=120, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.seq6 = nn.Sequential(
            nn.Conv2d(in_channels=184, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.seq7 = nn.Sequential(
            nn.Conv2d(in_channels=88, out_channels=16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        # self.conv_lstm = ConvLSTM(input_size=(128, 160),
        #                         input_dim=1,
        #                         hidden_dim=[128, 160, 1],
        #                         kernel_size=(3, 3),
        #                         num_layers=3,
        #                         batch_first=False,
        #                         bias=True,
        #                         return_all_layers=False,
        #                         device=device)

    def forward(self, input):
        seq1_out = self.seq1(input)
        down_sample1 = self.maxpool1(seq1_out)
        seq2_out = self.seq2(down_sample1)
        down_sample2 = self.maxpool2(torch.cat((down_sample1, seq2_out), dim=1))
        seq3_out = self.seq3(down_sample2)
        skip1 = torch.cat((down_sample2, seq3_out), dim=1)
        down_sample3 = self.maxpool3(skip1)
        seq4_out = self.seq4(down_sample3)
        skip2 = torch.cat((down_sample3, seq4_out), dim=1)
        seq5_out = self.seq5(self.maxpool4(skip2))
        up_sample1 = F.interpolate(seq5_out, scale_factor=2, mode='bilinear', align_corners=True)
        seq6_out = self.seq6(torch.cat((up_sample1, skip2), dim=1))
        up_sample2 = F.interpolate(seq6_out, scale_factor=2, mode='bilinear', align_corners=True)
        output = self.seq7(torch.cat((up_sample2, skip1), dim=1))
        # layer_output_list, _ = self.conv_lstm(output[None, ...])
        # layer_output = layer_output_list[0].reshape(output.shape)
        # return output, layer_output
        return output