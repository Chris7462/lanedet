"""
Custom ConvModule to replace mmcv.cnn.ConvModule
Minimal implementation with Conv2d + BatchNorm + Activation
"""

import torch.nn as nn


class ConvModule(nn.Module):
    """A Conv2d module with optional batch normalization and activation.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple): Stride of the convolution. Default: 1
        padding (int or tuple): Padding added to all sides of the input. Default: 0
        dilation (int or tuple): Spacing between kernel elements. Default: 1
        groups (int): Number of blocked connections. Default: 1
        bias (bool or str): If True, adds a learnable bias. Default: 'auto'
        conv_cfg (dict): Config for convolution layer. Default: None
        norm_cfg (dict): Config for normalization layer. Default: None
        act_cfg (dict): Config for activation layer. Default: dict(type='ReLU')
    """
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 inplace=True,
                 order=('conv', 'norm', 'act')):
        super(ConvModule, self).__init__()
        
        # Determine bias
        if bias == 'auto':
            bias = False if norm_cfg else True
        
        # Build convolution layer
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        
        # Build normalization layer
        self.norm = None
        if norm_cfg is not None:
            norm_type = norm_cfg.get('type', 'BN')
            if norm_type == 'BN':
                self.norm = nn.BatchNorm2d(out_channels)
            elif norm_type == 'GN':
                num_groups = norm_cfg.get('num_groups', 32)
                self.norm = nn.GroupNorm(num_groups, out_channels)
            elif norm_type == 'LN':
                self.norm = nn.LayerNorm(out_channels)
            # Add other norm types as needed
        
        # Build activation layer
        self.activate = None
        if act_cfg is not None:
            act_type = act_cfg.get('type', 'ReLU')
            if act_type == 'ReLU':
                self.activate = nn.ReLU(inplace=inplace)
            elif act_type == 'LeakyReLU':
                negative_slope = act_cfg.get('negative_slope', 0.01)
                self.activate = nn.LeakyReLU(negative_slope, inplace=inplace)
            elif act_type == 'PReLU':
                self.activate = nn.PReLU()
            elif act_type == 'Sigmoid':
                self.activate = nn.Sigmoid()
            elif act_type == 'Tanh':
                self.activate = nn.Tanh()
            # Add other activation types as needed
        
        self.order = order
    
    def forward(self, x):
        for layer_name in self.order:
            if layer_name == 'conv':
                x = self.conv(x)
            elif layer_name == 'norm' and self.norm is not None:
                x = self.norm(x)
            elif layer_name == 'act' and self.activate is not None:
                x = self.activate(x)
        return x
