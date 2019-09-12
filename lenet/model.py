import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):

    def __init__(self, channel_num):

        super(BasicBlock, self).__init__()

        self.blockConv = nn.Sequential(nn.Conv2d(channel_num,channel_num,kernel_size=5,stride=1,padding=2),
                                       nn.BatchNorm2d(channel_num),
                                       nn.ReLU(),
                                       nn.Conv2d(channel_num,channel_num,kernel_size=5,stride=1,padding=2),
                                       nn.BatchNorm2d(channel_num))

    def forward(self, x):

        out = self.blockConv(x) + x
        return F.relu(out)

class My_LeNET(nn.Module):

    def __init__(self):

        super(My_LeNET, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))

        self.block = nn.ModuleList([BasicBlock(32) for _ in range(10)])

        self.poolAfterBlock = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
                                    nn.ReLU())
        
        self.fc = nn.Sequential(nn.Linear(in_features=64*14*14, out_features=64*14*14),
                                nn.ReLU(),
                                nn.Dropout(p=0.4),
                                nn.Linear(in_features=64*14*14, out_features=2),
                                nn.LogSoftmax(dim=1)
                                )

    def forward(self, x):

        out = self.conv1(x)
        
        for i in range(10):
            out = self.block[i](out)

        out = self.poolAfterBlock(out)

        out = self.conv2(out)

        out = out.view(x.size()[0], -1)

        out = self.fc(out)

        return out

        



