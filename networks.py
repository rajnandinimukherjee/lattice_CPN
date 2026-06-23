import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F


class AlphaNet(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim=16):
        super(AlphaNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, dtype=torch.double),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, dtype=torch.double),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, dtype=torch.double)
        )
        self.hd = hidden_dim

    def forward(self, X):
        return self.network(X)


class DoubleConv(nn.Module):
    """(conv => ReLU => conv => ReLU) block"""

    def __init__(self, input_dim, output_dim, last_relu=True):
        super().__init__()
        layers = [
            nn.Conv2d(input_dim, output_dim, kernel_size=3,
                      padding=1, padding_mode="circular",
                      dtype=torch.double),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_dim, output_dim, kernel_size=3,
                      padding=1, padding_mode="circular",
                      dtype=torch.double)
        ]

        # Conditionally add the last ReLU
        if last_relu:
            layers.append(nn.ReLU(inplace=True))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class EncBlock(nn.Module):
    """ Encoder Block - down sampling """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        # (input_dim,H,W) -> (output_dim,H,W)
        self.conv = DoubleConv(input_dim, output_dim)
        # (output_dim,H,W) -> (output_dim,H/2,W/2)
        self.down = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.down(x)

        return x, p


class DecBlock(nn.Module):
    """ Decoder Block - up sampling"""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            # (input_dim,H,W) -> (output_dim, 2H, 2W)
            in_channels=input_dim, out_channels=output_dim,
            kernel_size=2, stride=2, padding=0, dtype=torch.double)
        self.conv = DoubleConv(2*output_dim, output_dim)

    def forward(self, inputs, to_concat):
        x = self.upsample(inputs)
        x = torch.cat([x, to_concat], dim=1)
        x = self.conv(x)

        return x


class UNet(nn.Module):

    def __init__(self, input_dim, output_dim, hd=None):
        super().__init__()
        hd = hd or input_dim*2
        self.enc1 = EncBlock(input_dim, hd)
        self.enc2 = EncBlock(hd, 2*hd)
        self.enc3 = EncBlock(2*hd, 4*hd)

        self.bottleneck = DoubleConv(4*hd, 8*hd)

        self.dec1 = DecBlock(8*hd, 4*hd)
        self.dec2 = DecBlock(4*hd, 2*hd)
        self.dec3 = DecBlock(2*hd, hd)

        self.output = nn.Conv2d(hd, output_dim, kernel_size=3,
                                padding_mode="circular", padding=1,
                                dtype=torch.double)

    def forward(self, X):
        x = X.permute(2, 0, 1).unsqueeze(0)
        # Down
        lvl1, d1 = self.enc1(x)
        lvl2, d2 = self.enc2(d1)
        lvl3, d3 = self.enc3(d2)

        # Bottleneck
        bn = self.bottleneck(d3)

        # Up
        u1 = self.dec1(bn, lvl3)
        u2 = self.dec2(u1, lvl2)
        u3 = self.dec3(u2, lvl1)

        out = self.output(u3).squeeze(0).permute(1, 2, 0)
        return out
