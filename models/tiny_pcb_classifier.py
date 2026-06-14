import torch
from torch import nn


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class TinyPCBClassifier96(nn.Module):
    """Small STM32-friendly PCB defect classifier.

    Uses only Conv2d, BatchNorm2d, ReLU, AdaptiveAvgPool2d, and Linear.
    Expected input shape is [N, 3, 96, 96], output shape is [N, num_classes].
    """

    def __init__(self, num_classes: int = 6) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")

        self.features = nn.Sequential(
            ConvBNReLU(3, 16, stride=2),    # 48x48
            ConvBNReLU(16, 16, stride=1),
            ConvBNReLU(16, 24, stride=2),   # 24x24
            ConvBNReLU(24, 24, stride=1),
            ConvBNReLU(24, 32, stride=2),   # 12x12
            ConvBNReLU(32, 32, stride=1),
            ConvBNReLU(32, 48, stride=2),   # 6x6
            ConvBNReLU(48, 48, stride=1),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(48, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_tiny_pcb_classifier_96(num_classes: int = 6) -> nn.Module:
    return TinyPCBClassifier96(num_classes=num_classes)


if __name__ == "__main__":
    model = build_tiny_pcb_classifier_96(num_classes=6)
    model.eval()
    x = torch.randn(1, 3, 96, 96)
    with torch.no_grad():
        y = model(x)
    print(f"Input shape: {tuple(x.shape)}")
    print(f"Output shape: {tuple(y.shape)}")
    print(f"Trainable parameters: {count_parameters(model)}")
