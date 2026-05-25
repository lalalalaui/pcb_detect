import torch
from torch import nn


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvAutoEncoder(nn.Module):
    """Convolutional autoencoder for 3x128x128 PCB patches."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(3, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 128 -> 64
            ConvBlock(32, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64 -> 32
            ConvBlock(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32 -> 16
            ConvBlock(128, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16 -> 8
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # 8 -> 16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ConvBlock(128, 128),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # 16 -> 32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ConvBlock(64, 64),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # 32 -> 64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ConvBlock(32, 32),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),  # 64 -> 128
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        recon = self.decoder(encoded)
        return recon


if __name__ == "__main__":
    model = ConvAutoEncoder()
    x = torch.rand(2, 3, 128, 128)
    with torch.no_grad():
        recon = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {recon.shape}")
    print(f"Parameter count: {count_parameters(model)}")
    print(f"Input/output shape match: {x.shape == recon.shape}")
