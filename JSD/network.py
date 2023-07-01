import torch
import torch.nn as nn

class UnetClassic3D(nn.Module):
    def __init__(self, n_classes):
        super(UnetClassic3D, self).__init__()
        self.n_classes = n_classes

        self.contracting_block1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=[3, 3, 3], padding=1),
            nn.Conv3d(32, 32, kernel_size=[3, 3, 3], padding=1),
        )

        self.contracting_block2 = nn.Sequential(
            nn.MaxPool3d([2, 2, 2]),
            nn.Conv3d(32, 64, kernel_size=[3, 3, 3], padding=1),
            nn.Conv3d(64, 64, kernel_size=[3, 3, 3], padding=1),
        )

        self.contracting_block3 = nn.Sequential(
            nn.MaxPool3d([2, 2, 2]),
            nn.Conv3d(64, 128, kernel_size=[3, 3, 3], padding=1),
            nn.Conv3d(128, 64, kernel_size=[3, 3, 3], padding=1),
        )


        self.expanding_block1 = nn.Sequential(
            nn.Conv3d(96, 64, kernel_size=[3, 3, 3], padding=1),
            nn.Conv3d(64, 64, kernel_size=[3, 3, 3], padding=1),
            nn.Conv3d(64, 3 * n_classes, kernel_size=[3, 3, 3], padding=1),
        )

        self.expanding_block2 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=[3, 3, 3], padding=1),
            nn.Conv3d(128, 64, kernel_size=[3, 3, 3], padding=1),
            nn.Upsample(scale_factor=2),
        )

        self.expanding_block3 = nn.Upsample(scale_factor=2)


        self.contracting_block11 = nn.Sequential(
            nn.Conv3d(1 + 3 * n_classes, 32, kernel_size=[3, 3, 3], padding=1),
            nn.Conv3d(32, 32, kernel_size=[3, 3, 3], padding=1),
        )

        self.contracting_block21 = nn.Sequential(
            nn.MaxPool3d([2, 2, 2]),
            nn.Conv3d(32, 64, kernel_size=[3, 3, 3], padding=1),
            nn.Conv3d(64, 64, kernel_size=[3, 3, 3], padding=1),
        )

        self.contracting_block31 = nn.Sequential(
            nn.MaxPool3d([2, 2, 2]),
            nn.Conv3d(64, 128, kernel_size=[3, 3, 3], padding=1),
            nn.Conv3d(128, 64, kernel_size=[3, 3, 3], padding=1),
        )


        self.expanding_block11 = nn.Sequential(
            nn.Conv3d(96, 64, kernel_size=[3, 3, 3], padding=1),
            nn.Conv3d(64, 64, kernel_size=[3, 3, 3], padding=1),
            nn.Conv3d(64, self.n_classes, kernel_size=[3, 3, 3], padding=1),
        )

        self.expanding_block21 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=[3, 3, 3], padding=1),
            nn.Conv3d(128, 64, kernel_size=[3, 3, 3], padding=1),
            nn.Upsample(scale_factor=2),
        )

        self.expanding_block31 = nn.Upsample(scale_factor=2)

        self.fc = nn.Linear(256*256*8, 3)

    def forward(self, x):
        x1 = self.contracting_block1(x)
        x2 = self.contracting_block2(x1)
        x3 = self.contracting_block3(x2)

        x3 = self.expanding_block3(x3)
        x2 = self.expanding_block2(torch.cat([x2, x3], dim=1))
        x1 = self.expanding_block1(torch.cat([x1, x2], dim=1))

        x11 = self.contracting_block11(torch.cat([x, x1], dim=1))
        x21 = self.contracting_block21(x11)
        x31 = self.contracting_block31(x21)

        x31 = self.expanding_block31(x31)
        x21 = self.expanding_block21(torch.cat([x21, x31], dim=1))
        x11 = self.expanding_block11(torch.cat([x11, x21], dim=1))

        x11 = x11.view(x11.shape[0], self.n_classes, -1)
        x = self.fc(x11)
        x = torch.sigmoid(x)

        return x
