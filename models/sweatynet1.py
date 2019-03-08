import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.ConvLSTM import ConvLSTM

class SweatyNet1(nn.Module):
    def __init__(self, input_size, device, use_ConvLSTM=False, seq_len=2):
        super(SweatyNet1, self).__init__()
        self.height, self.width, self.channels = input_size
        self.out_height, self.out_width = int(self.height/4), int(self.width/4)
        self.use_ConvLSTM = use_ConvLSTM
        self.device = device
        self.seq_len = seq_len
        self.seq1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.seq2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.seq3 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
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
        # Add Conv LSTM layer if flag is True
        if self.use_ConvLSTM:
            self.conv_lstm = ConvLSTM(input_size=(self.out_height, self.out_width),
                                    input_dim=1,
                                    hidden_dim=[32, 16, 1],
                                    kernel_size=(3, 3),
                                    num_layers=3,
                                    batch_first=True,
                                    bias=True,
                                    return_all_layers=False,
                                    device=device)
            self.seq8 = nn.Sequential(
                nn.BatchNorm2d(1),
                nn.ReLU()
            )

    def forward(self, input, prev_frame):
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
        seq7_out = self.seq7(torch.cat((up_sample2, skip1), dim=1))
        conv_lstm_in = None
        if self.use_ConvLSTM:
            for idx in range(seq7_out.shape[0]):
                # Create sequence of frames
                if prev_frame is None:
                    seq_conv_lstm = seq7_out[idx][None, ...]
                    # For very first frame in epoch just copy it as per sequence length
                    seq_conv_lstm = seq_conv_lstm.repeat((self.seq_len, 1, 1, 1))
                else:
                    seq_conv_lstm = torch.cat((prev_frame[1:], seq7_out[idx][None, ...]))
                
                if conv_lstm_in is None:
                    conv_lstm_in = seq_conv_lstm[None, ...]
                else:                    
                    conv_lstm_in = torch.cat((conv_lstm_in, seq_conv_lstm[None, ...]))                    
            conv_lstm_list, _ = self.conv_lstm(conv_lstm_in)
            conv_lstm_out = conv_lstm_list[0][:, -1, :, :]
            seq8_out = self.seq8(conv_lstm_out)
            output = seq8_out
        else:
            output = seq7_out
        
        if self.use_ConvLSTM:
            return output, seq_conv_lstm
        else:
            return output, None