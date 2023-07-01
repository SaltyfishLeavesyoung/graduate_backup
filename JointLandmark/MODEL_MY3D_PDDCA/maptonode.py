import torch
import torch.nn as nn

class MapToNode(nn.Module):
    """Base MapToNode, only using feat4"""
    def __init__(self, in_channels, num_points):
        super(MapToNode, self).__init__()
        self.num_points = num_points
        self.conv_out = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        self.conv_to_node = nn.Sequential(
            nn.Conv2d(256, num_points*4, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_points*4),
            nn.ReLU(inplace=True))
        '''self.conv2node = nn.Sequential(
            nn.Conv3d(in_channels, num_points*4, kernel_size=3, padding=1, stride=1),
            #nn.BatchNorm3d(num_points*4),
            nn.ReLU(inplace=True))'''

    def forward(self, x):
        x = self.conv_out(x)
        x = self.conv_to_node(x)
        # x = self.conv2node(x)
        x = x.view(x.size(0), self.num_points, -1)

        return x

if __name__=='__main__':
    x= torch.randn(2,512,25,20)
    mtn = MapToNode(512,19)
    out=mtn(x)
    print(out.size())
