import torch
from torch import nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


def build_mobilenet_v2(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Build a MobileNetV2 classifier for PCB defect classes."""
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")

    weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


if __name__ == "__main__":
    model = build_mobilenet_v2(num_classes=6, pretrained=False)
    model.eval()

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        y = model(x)

    print(f"MobileNetV2 output shape: {y.shape}")
