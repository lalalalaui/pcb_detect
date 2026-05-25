import torch
from torch import nn


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


class TinyAutoEncoder(nn.Module):
    """Small ONNX-friendly autoencoder for edge deployment.

    Works with inputs such as 3x64x64 and 3x96x96 because the model downsamples
    and upsamples by a total factor of 8.
    """

    def __init__(self, base_channels: int = 12) -> None:
        super().__init__()
        if base_channels <= 0:
            raise ValueError("base_channels must be positive.")

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.encoder = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(c3, c2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c2, c1, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c1, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        recon = self.decoder(encoded)
        return recon


if __name__ == "__main__":
    model = TinyAutoEncoder()
    for size in (64, 96):
        x = torch.rand(2, 3, size, size)
        with torch.no_grad():
            recon = model(x)

        print(f"Input shape: {x.shape}")
        print(f"Output shape: {recon.shape}")
        print(f"Input/output shape match: {x.shape == recon.shape}")

    print(f"Parameter count: {count_parameters(model)}")
