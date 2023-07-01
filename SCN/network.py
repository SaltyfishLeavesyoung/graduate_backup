import torch
import torch.nn as nn

class SCNetLocal(nn.Module):
    def __init__(self):
        super(SCNetLocal, self).__init__()

        self.contracting_block1 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=[3, 3, 3], padding=1),
            nn.Dropout(0.5),
            nn.Conv3d(64, 64, kernel_size=[3, 3, 3], padding=1),
        )

        self.contracting_block2 = nn.Sequential(
            nn.AvgPool3d([2, 2, 2]),
            nn.Conv3d(64, 64, kernel_size=[3, 3, 3], padding=1),
            nn.Dropout(0.5),
            nn.Conv3d(64, 64, kernel_size=[3, 3, 3], padding=1),
        )

        self.contracting_block3 = nn.Sequential(
            nn.AvgPool3d([2, 2, 2]),
            nn.Conv3d(64, 64, kernel_size=[3, 3, 3], padding=1),
            nn.Dropout(0.5),
            nn.Conv3d(64, 64, kernel_size=[3, 3, 3], padding=1),
        )

        self.contracting_block4 = nn.Sequential(
            nn.AvgPool3d([2, 2, 2]),
            nn.Conv3d(64, 64, kernel_size=[3, 3, 3], padding=1),
            nn.Dropout(0.5),
            nn.Conv3d(64, 64, kernel_size=[3, 3, 3], padding=1)
        )

        self.parallel_block1 = nn.Conv3d(128, 64, kernel_size=[3, 3, 3], padding=1)  
        self.parallel_block2 = nn.Conv3d(128, 64, kernel_size=[3, 3, 3], padding=1)
        self.parallel_block3 = nn.Conv3d(128, 64, kernel_size=[3, 3, 3], padding=1)
        self.parallel_block4 = nn.Conv3d(64, 64, kernel_size=[3, 3, 3], padding=1)

        self.expanding_block2 = nn.Upsample(scale_factor=2)
        self.expanding_block3 = nn.Upsample(scale_factor=2)
        self.expanding_block4 = nn.Upsample(scale_factor=2)

    def forward(self, x):
        # print(x.shape) torch.Size([1, 64, 8, 256, 256])
        x1 = self.contracting_block1(x)
        # print(x1.shape) torch.Size([1, 64, 8, 256, 256])
        x2 = self.contracting_block2(x1)
        # print(x2.shape) torch.Size([1, 64, 4, 128, 128])
        x3 = self.contracting_block3(x2)
        # print(x3.shape) torch.Size([1, 64, 2, 64, 64])
        x4 = self.contracting_block4(x3)
        # print(x4.shape) torch.Size([1, 64, 1, 32, 32])

        x4 = self.expanding_block4(self.parallel_block4(x4))
        x3 = self.expanding_block3(self.parallel_block3(torch.cat([x3, x4], dim=1)))
        x2 = self.expanding_block2(self.parallel_block2(torch.cat([x2, x3], dim=1)))
        x1 = self.parallel_block1(torch.cat([x1, x2], dim=1))

        return x1

class UnetClassic3D(nn.Module):
    def __init__(self):
        super(UnetClassic3D, self).__init__()

        self.contracting_block1 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=[3, 3, 3], padding=1),
            nn.Conv3d(64, 64, kernel_size=[3, 3, 3], padding=1),
        )

        self.contracting_block2 = nn.Sequential(
            nn.AvgPool3d([2, 2, 2]),
            nn.Conv3d(64, 64, kernel_size=[3, 3, 3], padding=1),
            nn.Conv3d(64, 64, kernel_size=[3, 3, 3], padding=1),
        )

        self.contracting_block3 = nn.Sequential(
            nn.AvgPool3d([2, 2, 2]),
            nn.Conv3d(64, 64, kernel_size=[3, 3, 3], padding=1),
            nn.Conv3d(64, 64, kernel_size=[3, 3, 3], padding=1),
        )

        self.contracting_block4 = nn.Sequential(
            nn.AvgPool3d([2, 2, 2]),
            nn.Conv3d(64, 64, kernel_size=[3, 3, 3], padding=1),
            nn.Conv3d(64, 64, kernel_size=[3, 3, 3], padding=1)
        )

        self.expanding_block2 = nn.Upsample([2,2,2])
        self.expanding_block3 = nn.Upsample([2,2,2])
        self.expanding_block4 = nn.Upsample([2,2,2])

    def forward(self, x):
        x1 = self.contracting_block1(x)
        x2 = self.contracting_block2(x1)
        x3 = self.contracting_block3(x2)
        x4 = self.contracting_block4(x3)

        x4 = self.expanding_block4(x4)
        x3 = self.expanding_block3(torch.cat([x3, x4], dim=1))
        x2 = self.expanding_block2(torch.cat([x2, x3], dim=1))
        x1 = torch.cat([x1, x2], dim=1)

        return x1

class network_scn(nn.Module):
    def __init__(self, num_heatmaps, factor=[8,8,8]):
        super(network_scn, self).__init__()
        self.num_heatmaps = num_heatmaps
        self.factor = factor

        self.layer1 = nn.Conv3d(1, 64, kernel_size=[3, 3, 3], padding=1)
        self.scnet_local = SCNetLocal()
        self.layer2 = nn.Conv3d(64, num_heatmaps, kernel_size=[3, 3, 3], padding=1)
        self.layer3 = nn.AvgPool3d(factor)
        self.layer4 = nn.Conv3d(num_heatmaps, 64, kernel_size=[7, 7, 7], padding=3)
        self.layer5 = nn.Conv3d(64, 64, kernel_size=[7, 7, 7], padding=3)
        self.layer6 = nn.Conv3d(64, num_heatmaps, kernel_size=[7, 7, 7], padding=3)
        self.layer7 = nn.Upsample(scale_factor=8)

        self.final = nn.Conv3d(num_heatmaps, num_heatmaps, kernel_size=1, padding=0)

        self.fc = nn.Linear(256*256*8, 3)

    def forward(self, x):
        x = self.layer1(x)
        x = self.scnet_local(x)
        local_heatmaps = self.layer2(x)
        # print(local_heatmaps.shape) torch.Size([1, 6, 8, 256, 256])
        x = self.layer3(local_heatmaps)
        # print(x.shape) torch.Size([1, 6, 1, 32, 32])
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        spatial_heatmaps = self.layer7(x)
        # heatmaps = local_heatmaps * spatial_heatmaps
        # return heatmaps, local_heatmaps, spatial_heatmaps
        
        x = self.final(spatial_heatmaps)
        x = x.view(x.size(0), self.num_heatmaps, -1)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x, x, x

class network_unet(nn.Module):
    def __init__(self, num_heatmaps):
        super(network_unet, self).__init__()
        self.num_heatmaps = num_heatmaps

        self.layer1 = nn.Conv3d(1, 64, kernel_size=[3, 3, 3])
        self.scnet_local = UnetClassic3D()
        self.layer2 = nn.Conv3d(64, num_heatmaps, kernel_size=[3, 3, 3])

    def forward(self, x):
        x = self.layer1(x)
        x = self.scnet_local(x)
        heatmaps = self.layer2(x)
        return heatmaps, heatmaps, heatmaps
