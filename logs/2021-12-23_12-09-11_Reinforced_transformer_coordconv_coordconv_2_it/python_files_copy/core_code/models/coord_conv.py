import torch


class CoordConv2d(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, 
                 padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True):
        super(CoordConv2d, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels = in_channels + 2, out_channels = out_channels, 
                                    kernel_size = kernel_size, stride = stride, padding = padding, 
                                    dilation = dilation, groups = groups, bias = bias)

    def add_coords(self, input):
        aux_vector = torch.range(-1, 1, 2/(input.shape[-1]-1), device = input.device)
        i_coord = aux_vector.unsqueeze(1).repeat(input.shape[0], 1, 1, input.shape[-1])
        j_coord = torch.transpose(input = i_coord, dim0 = -1, dim1 = -2)
        out = torch.cat([input, i_coord, j_coord], dim = 1)
        return out

    def forward(self, input):
        x = self.add_coords(input)
        out = self.conv(x)
        return out
