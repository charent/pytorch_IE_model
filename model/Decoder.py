from numpy.core.numeric import outer
import torch
import torch.nn as nn
from torch import Tensor

# 膨胀残差卷积
class DGCNNDecoder(nn.Module):
    def __init__(self, dilations: list, kernel_sizes: list, in_channels: int):
        '''
        dilation gate cnn decoder
        inputs: [batch_size, seq_len, in_channels]
        outs: [batch_szie, seq_len, out_channels]
        '''
        super(DGCNNDecoder, self).__init__()

        if len(dilations) != len(kernel_sizes):
            raise ValueError('length of dilations and kernel_sizes must be same.')
        
        self.dropout = nn.Dropout(p=0.1)
        self.in_channels = in_channels

        self.conv1d_layer = nn.ModuleList()
        for i in range(len(dilations)):
            self.conv1d_layer.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=kernel_sizes[i],
                        stride=1,
                        dilation=dilations[i],
                        padding=int((dilations[i] * (kernel_sizes[i] - 1)) / 2) # padding保证输入和和输出的维度一样
                    ),
                    nn.ReLU(),
                )
        )

    def forward(self, inputs: Tensor):
        
        inputs = inputs.permute(0, 2, 1)

        for conv in self.conv1d_layer:
            outs = conv(inputs)
            inputs = inputs + outs
        
        outs = inputs.permute(0, 2, 1)

        return outs

class DGCNNConcatDecoder(nn.Module):
    def __init__(self, dilations: list, kernel_sizes: list, in_channels: int, out_channels: int):
        '''
        dilation gate cnn Decoder
        inputs: [batch_size, seq_len, in_channels]
        outs: [batch_szie, seq_len, out_channels]
        '''
        super(DGCNNConcatDecoder, self).__init__()

        if len(dilations) != len(kernel_sizes):
            raise ValueError('length of dilations and kernel_sizes must be same.')
        
        self.conv1d_layer = nn.ModuleList()

        for i in range(len(dilations)):
            self.conv1d_layer.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_sizes[i],
                    stride=1,
                    dilation=dilations[i],
                    padding=int((dilations[i] * (kernel_sizes[i] - 1)) / 2) # padding保证输入和和输出的维度一样
                )
            )

    def forward(self, inputs: Tensor, res: Tensor=None):
        '''
        res: [batch_size, seq_len, out_channels]
        '''
        inputs = inputs.permute(0, 2, 1)

        cnn_outs = []
        for conv in self.conv1d_layer:
            # outs：[batch_size, seq_len, out_channels]
            outs = conv(inputs).permute(0, 2, 1)
            if res is not None:
               outs = res + outs
            cnn_outs.append(outs)
        
        outs = torch.cat(cnn_outs, dim=2)
        return outs

class DGCNNPairCatDecoder(nn.Module):
    def __init__(self, dilations: list, kernel_sizes: list, in_channels: int):
        '''
        dilation gate cnn Decoder
        inputs: [batch_size, seq_len, in_channels]
        outs: [batch_szie, seq_len, out_channels]
        '''
        super(DGCNNPairCatDecoder, self).__init__()

        if len(dilations) != len(kernel_sizes):
            raise ValueError('length of dilations and kernel_sizes must be same.')
        
        self.conv1d_layer = nn.ModuleList()

        for d, k in zip(dilations, kernel_sizes):
            assert len(d) == len(k)
            n = len(d)
            cnn_layer = nn.ModuleList()
            for i in range(n):
                cnn_layer.append(
                    nn.Sequential(
                        nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=k[i],
                            stride=1,
                            dilation=d[i],
                            padding=int((d[i] * (k[i] - 1)) / 2) # padding保证输入和和输出的维度一样
                        ),
                        nn.ReLU(),
                    )
            )
            self.conv1d_layer.append(cnn_layer)

    def forward(self, inputs: Tensor):
        '''
        share_feature: [batch_size, seq_len, out_channels]
        '''
        inputs = inputs.permute(0, 2, 1)

        cnn_outs = []
        for cnn_list in self.conv1d_layer:
            outs = inputs
            for conv in cnn_list:
                outs = conv(outs)
                outs = outs + inputs
            outs = outs.permute(0, 2, 1)
            cnn_outs.append(outs)
        
        outs = torch.cat(cnn_outs, dim=2)
        return outs

class LinearDecoder(nn.Module):
    def __init__(self, in_features: int, out_features: int, forward_dim: int, activation: nn.Module):
        super(LinearDecoder, self).__init__()

        self.Linear_layer = nn.Sequential(
            nn.Linear(in_features, forward_dim),
            activation,
            nn.Linear(forward_dim, out_features)
        )

    def forward(self, inputs: Tensor):
        outs = self.Linear_layer(inputs)
        
        return outs

if __name__ == '__main__':
    
    device = 'cpu'
    inputs = torch.randn((32, 23, 64)).to(device)
    share_feature = torch.randn((32, 23, 32)).to(device)
    d = [[1,1,1], [2,3,5]]
    k = [[3,5,7], [3,3,3]]

    Decoder = DGCNNPairCatDecoder(dilations=d, kernel_sizes=k, in_channels=64, device=device).to(device)
    out = Decoder(inputs)
    print(out.shape)

    exit()

    d = [1, 3, 5, 4, 2, 1]
    k = [5, 3, 3, 3, 5, 6]
    Decoder = DGCNNConcatDecoder(dilations=d, kernel_sizes=k, in_channels=64, out_channels=32, device=device).to(device)

    out = Decoder(inputs, share_feature)
    print(out.shape)
    
