from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlphaNet(nn.Module):
    """Small fully connected network for deformation parameters."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 16,
    ) -> None:
        """Build a three-layer multilayer perceptron."""
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, dtype=torch.float32),
        )
        self.hd = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map input features to deformation parameters."""
        return self.network(x)


class DoubleConv(nn.Module):
    """(conv => ReLU => conv => ReLU) block."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        ndim: int,
        last_relu: bool = True,
    ) -> None:
        """Build two circularly padded convolutions."""
        super().__init__()

        Conv = getattr(nn, f"Conv{ndim}d")

        layers = [
            Conv(
                input_dim,
                output_dim,
                kernel_size=3,
                padding=1,
                padding_mode="circular",
                dtype=torch.float32,
            ),
            nn.LeakyReLU(inplace=True),
            Conv(
                output_dim,
                output_dim,
                kernel_size=3,
                padding=1,
                padding_mode="circular",
                dtype=torch.float32,
            ),
        ]

        if last_relu:
            layers.append(nn.LeakyReLU(inplace=True))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply both convolutional layers."""
        return self.network(x)


class EncBlock(nn.Module):
    """Encoder block - down sampling."""

    def __init__(self, input_dim: int, output_dim: int, ndim: int) -> None:
        """Build the convolution and pooling stages."""
        super().__init__()

        self.conv = DoubleConv(input_dim, output_dim, ndim)
        self.down = getattr(nn, f"MaxPool{ndim}d")(2)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the skip features and down-sampled features."""
        x = self.conv(inputs)
        p = self.down(x)

        return x, p


class DecBlock(nn.Module):
    """Decoder block - up sampling."""

    def __init__(self, input_dim: int, output_dim: int, ndim: int) -> None:
        """Build the transpose convolution and feature merger."""
        super().__init__()

        ConvTranspose = getattr(nn, f"ConvTranspose{ndim}d")

        self.upsample = ConvTranspose(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=2,
            stride=2,
            padding=0,
            dtype=torch.float32,
        )

        self.conv = DoubleConv(2 * output_dim, output_dim, ndim)

    def forward(
        self,
        inputs: torch.Tensor,
        to_concat: torch.Tensor,
    ) -> torch.Tensor:
        """Upsample inputs and merge the matching encoder features."""
        x = self.upsample(inputs)
        target_shape = to_concat.shape[2:]
        current_shape = x.shape[2:]
        if current_shape != target_shape:
            # MaxPool rounds odd lattice lengths down. Restore the missing
            # periodic endpoint before concatenating the corresponding skip.
            padding = []
            for current, target in reversed(
                list(zip(current_shape, target_shape))
            ):
                if current > target or target - current > 1:
                    raise ValueError(
                        "cannot align decoder and encoder feature maps: "
                        f"{current_shape} vs {target_shape}"
                    )
                padding.extend((0, target - current))
            x = F.pad(x, tuple(padding), mode="circular")
        x = torch.cat([x, to_concat], dim=1)
        x = self.conv(x)

        return x


class UNet(nn.Module):
    """Dimension-independent U-Net with channels-last public inputs."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        ndim: int = 2,
        hd: int | None = None,
    ) -> None:
        """Build a U-Net for one, two, or three spatial dimensions."""
        super().__init__()

        if ndim not in (1, 2, 3):
            raise ValueError(f"ndim must be 1, 2, or 3, got {ndim}")
        self.ndim = ndim

        hd = hd or input_dim * 2

        self.enc1 = EncBlock(input_dim, hd, ndim)
        self.enc2 = EncBlock(hd, 2 * hd, ndim)
        self.enc3 = EncBlock(2 * hd, 4 * hd, ndim)

        self.bottleneck = DoubleConv(4 * hd, 8 * hd, ndim)

        self.dec1 = DecBlock(8 * hd, 4 * hd, ndim)
        self.dec2 = DecBlock(4 * hd, 2 * hd, ndim)
        self.dec3 = DecBlock(2 * hd, hd, ndim)

        Conv = getattr(nn, f"Conv{ndim}d")
        self.output = Conv(
            hd,
            output_dim,
            kernel_size=3,
            padding_mode="circular",
            padding=1,
            dtype=torch.float32,
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Predict a channels-last deformation field."""
        if X.ndim != self.ndim + 1:
            raise ValueError(f"expected {self.ndim} spatial axes and one channel axis")
        original_shape = X.shape[:-1]
        if any(length < 1 for length in original_shape):
            raise ValueError("spatial dimensions must be non-empty")
        # Three pooling stages require at least eight sites. For a smaller
        # periodic lattice, tile its input up to eight and crop the output
        # back to the physical lattice below.
        for dim, length in enumerate(original_shape):
            if length < 8:
                indices = torch.arange(8, device=X.device) % length
                X = torch.index_select(X, dim, indices)
        # Input: (... spatial dimensions ..., input_dim)
        # Move channels to the PyTorch position:
        #   (H, C)       -> (C, H)
        #   (H, W, C)    -> (C, H, W)
        #   (H, W, L, C) -> (C, H, W, L)
        x = X.movedim(-1, 0).unsqueeze(0)

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

        # Remove batch dimension and move channels back to the end
        out = self.output(u3).squeeze(0).movedim(0, -1)
        out = out[tuple(slice(0, length) for length in original_shape)]

        return out
