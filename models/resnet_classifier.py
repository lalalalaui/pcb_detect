import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_resnet18(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Build a ResNet18 classifier for PCB defect classes."""
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


if __name__ == "__main__":
    model = build_resnet18(num_classes=6, pretrained=False)
    model.eval()

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        y = model(x)

    print(f"ResNet18 output shape: {y.shape}")
